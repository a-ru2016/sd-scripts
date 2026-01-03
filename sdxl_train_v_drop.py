import argparse
import random
import time
import os
import sys
import math
from multiprocessing import Value
import numpy as np
import cv2
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
from library.device_utils import init_ipex, clean_memory_on_device
from library import deepspeed_utils, model_util, sdxl_model_util, sdxl_train_util, train_util, config_util
import train_network
from library.utils import setup_logging
from PIL import Image

init_ipex()
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Global variable for Embedding Dimension
DINO_EMBEDDING_DIM = 1024

# --- Monkey Patching for Dataset to load Mask and Visual Embed ---
original_getitem = train_util.DreamBoothDataset.__getitem__

def get_associated_files(image_path):
    # Infer mask and npy paths from image path
    # tools/prepare_v_drop_data.py generates:
    # image.png -> image_mask.png, image.npy
    base, ext = os.path.splitext(image_path)
    mask_path = f"{base}_mask.png"
    npy_path = f"{base}.npy"
    return mask_path, npy_path

def v_drop_getitem(self, index):
    # Call original getitem
    example = original_getitem(self, index)
    
    # Load extended data
    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
    image_index = self.buckets_indices[index].batch_index * bucket_batch_size
    
    masks = []
    visual_embeds = []
    
    for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
        image_info = self.image_data[image_key]
        mask_path, npy_path = get_associated_files(image_info.absolute_path)
        
        # Mask Loading
        if os.path.exists(mask_path):
            try:
                mask_img = Image.open(mask_path).convert("L")
                # Resize to match bucket resolution (before crop)
                # image_info.resized_size is (w, h)
                target_size = image_info.resized_size 
                mask_img = mask_img.resize(target_size, Image.NEAREST)
                
                # Crop based on how the original image was cropped
                c_top, c_left = example["crop_top_lefts"][i]
                h, w = example["target_sizes_hw"][i]
                
                mask_arr = np.array(mask_img)
                
                # Apply crop
                mask_arr = mask_arr[c_top : c_top + h, c_left : c_left + w]
                
                if mask_arr.shape[0] != h or mask_arr.shape[1] != w:
                    # Fallback for size mismatch
                    masks.append(torch.ones((1, h, w), dtype=torch.float32))
                else:
                    # Convert to tensor 0.0 - 1.0
                    mask_tensor = torch.from_numpy(mask_arr).float() / 255.0
                    masks.append(mask_tensor.unsqueeze(0)) # (1, H, W)
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                h, w = example["target_sizes_hw"][i]
                masks.append(torch.ones((1, h, w), dtype=torch.float32))
        else:
            # No mask found: use all ones (train on whole image)
            h, w = example["target_sizes_hw"][i]
            masks.append(torch.ones((1, h, w), dtype=torch.float32))

        # Visual Embed Loading
        if os.path.exists(npy_path):
            try:
                ve = np.load(npy_path)
                # Ensure shape is (DINO_EMBEDDING_DIM,)
                if ve.ndim == 2: ve = ve.flatten()
                
                if ve.shape[0] != DINO_EMBEDDING_DIM:
                    logger.warning(f"Dimension mismatch for {npy_path}: expected {DINO_EMBEDDING_DIM}, got {ve.shape[0]}. Using zeros.")
                    visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())
                else:
                    visual_embeds.append(torch.from_numpy(ve).float())
            except Exception as e:
                logger.warning(f"Error loading npy {npy_path}: {e}")
                visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())
        else:
            # No embedding found: use zeros
            visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())

    example["masks"] = torch.stack(masks) # (B, 1, H, W)
    example["visual_embeds"] = torch.stack(visual_embeds) # (B, Dim)
    
    return example

# Apply Patch
train_util.DreamBoothDataset.__getitem__ = v_drop_getitem
train_util.FineTuningDataset.__getitem__ = v_drop_getitem

# -----------------------------------------------------------

class SdxlVDropTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        sdxl_train_util.verify_sdxl_training_args(args)
        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset, weight_dtype):
        if args.cache_text_encoder_outputs:
             # V-Drop requires dynamic text embedding modification (Visual Injection),
             # so standard caching might conflict if it caches the final embeddings.
             # However, if we modify the embeddings *after* loading from cache, it might work,
             # provided we have access to the cached tensors in the training loop.
             pass
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, tokenizers, text_encoders, dataset, weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"].to(text_encoders[0].device)
            input_ids2 = batch["input_ids2"].to(text_encoders[0].device)
            
            with torch.enable_grad():
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

        encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device)
        encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device)
        pool2 = pool2.to(accelerator.device)

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        # Handle FP8
        input_dtype = weight_dtype
        
        noisy_latents = noisy_latents.to(input_dtype)
        
        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(input_dtype)

        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        
        # --- Visual Injection ---
        # Concatenate text embeddings
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(input_dtype)
        
        # Access the network (Projector)
        # Assuming self.network has been set in train()
        if hasattr(self, "network") and hasattr(self.network, "forward_projector"):
            visual_embeds = batch["visual_embeds"].to(accelerator.device).to(weight_dtype)
            
            # Visual Dropout check
            if not getattr(self, "visual_dropout_active", False):
                # Projector output: (B, 1, Dim)
                visual_tokens = self.network.forward_projector(visual_embeds) 
                
                # Replace trigger token with visual token
                input_ids = batch["input_ids2"]
                obj_0_token_id = self.obj_0_token_id
                
                # Iterate over batch
                for i in range(input_ids.shape[0]):
                    # Find indices of trigger token
                    indices = (input_ids[i] == obj_0_token_id).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        idx = indices[0] # Take the first occurrence
                        # Replace text embedding with projected visual embedding
                        # visual_tokens[i, 0, :] shape is (Dim)
                        text_embedding[i, idx, :] = visual_tokens[i, 0, :].to(dtype=text_embedding.dtype)
        
        vector_embedding = torch.cat([pool2.to(input_dtype), embs], dim=1).to(input_dtype)
        
        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def train(self, args):
        global DINO_EMBEDDING_DIM
        DINO_EMBEDDING_DIM = args.dino_embedding_dim
        logger.info(f"Setting DINO embedding dimension to {DINO_EMBEDDING_DIM}")

        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)
        
        # Tokenizer
        tokenizer = self.load_tokenizer(args) # list
        
        # Add trigger token
        trigger_token = getattr(args, "trigger_token", "[obj_0]")
        if trigger_token not in tokenizer[1].get_vocab():
            tokenizer[1].add_tokens([trigger_token])
            tokenizer[0].add_tokens([trigger_token])
        
        self.obj_0_token_id = tokenizer[1].convert_tokens_to_ids(trigger_token)
        logger.info(f"Trigger token: {trigger_token}, ID: {self.obj_0_token_id}")

        # Dataset
        blueprint_generator = config_util.BlueprintGenerator(config_util.ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
             user_config = config_util.load_user_config(args.dataset_config)
        else:
             subset = {"image_dir": args.train_data_dir}
             if args.in_json:
                 subset["metadata_file"] = args.in_json
             user_config = {
                "datasets": [{"subsets": [subset]}]
            }
        
        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        collator = train_util.collator_class(current_epoch, current_step, train_dataset_group)

        # Accelerator
        accelerator = train_util.prepare_accelerator(args)
        weight_dtype, _ = train_util.prepare_dtype(args)
        
        # Model
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # Enable xformers / mem_eff_attn
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # FP8 Base
        unet_weight_dtype = te_weight_dtype = weight_dtype
        if args.fp8_base:
            accelerator.print("enable fp8 training.")
            unet_weight_dtype = torch.float8_e4m3fn
            te_weight_dtype = torch.float8_e4m3fn

        unet.requires_grad_(False)
        
        # VAE handling
        vae_dtype = weight_dtype
        if args.no_half_vae:
            vae_dtype = torch.float32
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.eval()
        
        # Resize Token Embeddings
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        for te, tok in zip(text_encoders, tokenizer):
            te.resize_token_embeddings(len(tok))
            te.requires_grad_(False)
            if te.device.type != "cpu":
                if hasattr(te, "text_model") and hasattr(te.text_model, "embeddings"):
                     te.text_model.embeddings.to(dtype=weight_dtype)

        # Network
        import importlib
        network_module = importlib.import_module(args.network_module)
        network = network_module.create_network(
            1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet, 
            neuron_dropout=args.network_dropout,
            **args.network_args_dict if hasattr(args, "network_args_dict") else {}
        )
        network.apply_to(text_encoder, unet, True, True)
        self.network = network

        if args.gradient_checkpointing:
            unet.train()
            unet.enable_gradient_checkpointing()
            for te in text_encoders:
                te.train()
                te.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()
        else:
            unet.eval()
            for te in text_encoders:
                te.eval()

        # Optimizer
        trainable_params, _ = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        
        # Add Projector params to optimizer
        if hasattr(network, "projector"):
            trainable_params.append({"params": network.projector.parameters(), "lr": args.learning_rate})
        
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=args.max_data_loader_n_workers or 4
        )

        # Scheduler
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # Prepare
        if args.full_fp16:
             unet.to(weight_dtype)

        unet.to(accelerator.device)

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )

        # Noise Scheduler
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        
        # Loop
        num_train_epochs = args.max_train_epochs
        if num_train_epochs is None:
            num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
        global_step = 0
        
        logger.info(f"Start training for {num_train_epochs} epochs")
        
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        for epoch in range(num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(network):
                    # Visual Dropout Logic (Curriculum)
                    # p = t / max_steps * p_max
                    p_max = getattr(args, "visual_dropout_rate", 0.0)
                    
                    if args.max_train_steps is not None and args.max_train_steps > 0:
                        current_progress = global_step / args.max_train_steps
                    else:
                        total_steps_approx = len(train_dataloader) * num_train_epochs
                        current_progress = global_step / total_steps_approx if total_steps_approx > 0 else 0
                    
                    current_progress = min(max(current_progress, 0.0), 1.0)
                    p_drop = current_progress * p_max
                    self.visual_dropout_active = random.random() < p_drop
                    
                    # Forward
                    with torch.set_grad_enabled(True):
                        # Latents
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device).to(weight_dtype)
                        else:
                            # VAE encode
                            vae_dtype = weight_dtype
                            if args.no_half_vae:
                                vae_dtype = torch.float32

                            if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                                with torch.no_grad():
                                    latents = vae.encode(batch["images"].to(accelerator.device).to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)
                            else:
                                chunks = [batch["images"][i:i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)]
                                list_latents = []
                                for chunk in chunks:
                                    with torch.no_grad():
                                        list_latents.append(vae.encode(chunk.to(accelerator.device).to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype))
                                latents = torch.cat(list_latents, dim=0)
                            
                            if torch.any(torch.isnan(latents)):
                                logger.warning("NaN found in latents, replacing with zeros")
                                latents = torch.nan_to_num(latents, 0, out=latents)
                            
                            latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                        # Noise
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                        
                        # Get Text Cond
                        text_conds = self.get_text_cond(args, accelerator, batch, tokenizer, text_encoders, weight_dtype)
                        
                        # UNet Call (includes Visual Injection)
                        with accelerator.autocast():
                            noise_pred = self.call_unet(args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype)
                        
                        # Loss
                        if args.masked_loss:
                            masks = batch["masks"].to(accelerator.device).to(weight_dtype)
                            # Resize mask to latent size
                            masks = torch.nn.functional.interpolate(masks, size=latents.shape[2:], mode='nearest')
                            
                            # Masked Loss Weighting
                            # Loss = MSE * (1 + lambda * mask)
                            lambda_val = getattr(args, "masked_loss_lambda", 5.0)
                            weight = 1.0 + lambda_val * masks
                            
                            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                            loss = loss * weight
                            loss = loss.mean()
                        else:
                            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                            
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    progress_bar.set_postfix({"loss": loss.item()})
                
                if args.max_train_steps is not None and global_step >= args.max_train_steps: break
            if args.max_train_steps is not None and global_step >= args.max_train_steps: break

        progress_bar.close()

        # Save
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
             network = accelerator.unwrap_model(network)
             network.save_weights(os.path.join(args.output_dir, f"{args.output_name}.safetensors"), weight_dtype, None)

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--trigger_token", type=str, default="[obj_0]", help="Trigger token to replace with visual embedding")
    parser.add_argument("--visual_dropout_rate", type=float, default=0.0, help="Visual dropout rate")
    parser.add_argument("--masked_loss_lambda", type=float, default=5.0, help="Weight for masked region in loss")
    parser.add_argument("--dino_embedding_dim", type=int, default=1024, help="Embedding dimension of DINOv2 model (default 1024 for vitl14)")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlVDropTrainer()
    trainer.train(args)