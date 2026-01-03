import argparse
import random
import time
import os
import sys
import math
from multiprocessing import Value
import numpy as np
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
from library.device_utils import init_ipex, clean_memory_on_device
from library import deepspeed_utils, model_util, sdxl_model_util, sdxl_train_util, train_util, config_util
import train_network_ABM as train_network
from library.utils import setup_logging
from PIL import Image

init_ipex()
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Global variable for Embedding Dimension
DINO_EMBEDDING_DIM = 1024

# --- Monkey Patching for Dataset to load Mask and Visual Embed ---
original_getitem_db = train_util.DreamBoothDataset.__getitem__
original_getitem_ft = train_util.FineTuningDataset.__getitem__

def get_associated_files(image_path):
    base, ext = os.path.splitext(image_path)
    mask_path = f"{base}_mask.png"
    npy_path = f"{base}.npy"
    return mask_path, npy_path

def v_drop_getitem_logic(self, index, example, args):
    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
    image_index = self.buckets_indices[index].batch_index * bucket_batch_size
    
    alpha_masks = []
    visual_embeds = []
    
    lambda_val = getattr(args, "masked_loss_lambda", 1.0)
    
    for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
        image_info = self.image_data[image_key]
        mask_path, npy_path = get_associated_files(image_info.absolute_path)
        
        h, w = example["target_sizes_hw"][i]
        mask_tensor = torch.zeros((h, w), dtype=torch.float32)

        if os.path.exists(mask_path):
            try:
                mask_img = Image.open(mask_path).convert("L")
                target_size = image_info.resized_size 
                mask_img = mask_img.resize(target_size, Image.NEAREST)
                
                c_top, c_left = example["crop_top_lefts"][i]
                mask_arr = np.array(mask_img)
                mask_arr = mask_arr[c_top : c_top + h, c_left : c_left + w]
                
                if mask_arr.shape[0] == h and mask_arr.shape[1] == w:
                    mask_tensor = torch.from_numpy(mask_arr).float() / 255.0
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
        
        weighted_mask = 1.0 + lambda_val * mask_tensor
        alpha_masks.append(weighted_mask)

        if os.path.exists(npy_path):
            try:
                ve = np.load(npy_path)
                if ve.ndim == 2: ve = ve.flatten()
                if ve.shape[0] != DINO_EMBEDDING_DIM:
                    visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())
                else:
                    visual_embeds.append(torch.from_numpy(ve).float())
            except Exception as e:
                logger.warning(f"Error loading npy {npy_path}: {e}")
                visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())
        else:
            visual_embeds.append(torch.zeros(DINO_EMBEDDING_DIM).float())

    example["alpha_masks"] = torch.stack(alpha_masks) # (B, H, W)
    example["visual_embeds"] = torch.stack(visual_embeds) # (B, Dim)
    
    return example

def v_drop_getitem_db(self, index):
    example = original_getitem_db(self, index)
    return v_drop_getitem_logic(self, index, example, self.v_drop_args)

def v_drop_getitem_ft(self, index):
    example = original_getitem_ft(self, index)
    return v_drop_getitem_logic(self, index, example, self.v_drop_args)

# Apply monkey patch at module level so it persists in spawned worker processes
train_util.DreamBoothDataset.__getitem__ = v_drop_getitem_db
train_util.FineTuningDataset.__getitem__ = v_drop_getitem_ft

# --- Monkey Patching for FFValidationDataset ---
import library.fast_forward as fast_forward_module
original_getitem_ff = fast_forward_module.FFValidationDataset.__getitem__

def v_drop_getitem_ff(self, index):
    data = original_getitem_ff(self, index)
    
    img_path = self.img_paths[index]
    _, npy_path = get_associated_files(img_path)
    
    if os.path.exists(npy_path):
        try:
            ve = np.load(npy_path)
            if ve.ndim == 2: ve = ve.flatten()
            if ve.shape[0] != DINO_EMBEDDING_DIM:
                visual_embed = torch.zeros(DINO_EMBEDDING_DIM).float()
            else:
                visual_embed = torch.from_numpy(ve).float()
        except Exception as e:
            logger.warning(f"Error loading npy {npy_path}: {e}")
            visual_embed = torch.zeros(DINO_EMBEDDING_DIM).float()
    else:
        visual_embed = torch.zeros(DINO_EMBEDDING_DIM).float()
        
    data["visual_embeds"] = visual_embed
    return data

fast_forward_module.FFValidationDataset.__getitem__ = v_drop_getitem_ff

# -----------------------------------------------------------

class SdxlVDropTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.current_step_val = None
        self.network_captured = None

    def assert_extra_args(self, args, train_dataset_group):
        sdxl_train_util.verify_sdxl_training_args(args)
        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"
        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs"
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
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"].to(text_encoders[0].device)
            input_ids2 = batch["input_ids2"].to(text_encoders[0].device)
            
            with torch.set_grad_enabled(not args.network_train_unet_only):
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

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        input_dtype = weight_dtype
        noisy_latents = noisy_latents.to(input_dtype)
        
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(input_dtype)

        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(input_dtype)
        
        # --- Visual Injection & Dropout Logic ---
        if self.network_captured is not None:
            network = accelerator.unwrap_model(self.network_captured)
            if hasattr(network, "forward_projector"):
                visual_embeds = batch["visual_embeds"].to(accelerator.device).to(input_dtype)
                
                # Dropout Calculation
                p_max = getattr(args, "visual_dropout_rate", 0.0)
                if p_max > 0:
                    global_step = self.current_step_val.value if self.current_step_val is not None else 0
                    max_steps = args.max_train_steps if args.max_train_steps is not None else 1
                    current_progress = min(max(global_step / max_steps, 0.0), 1.0)
                    p_drop = current_progress * p_max
                    dropout_active = random.random() < p_drop
                else:
                    dropout_active = False

                if not dropout_active:
                    visual_tokens = network.forward_projector(visual_embeds) 
                    input_ids = batch["input_ids2"]
                    obj_0_token_id = self.obj_0_token_id
                    
                    for i in range(input_ids.shape[0]):
                        indices = (input_ids[i] == obj_0_token_id).nonzero(as_tuple=True)[0]
                        if len(indices) > 0:
                            idx = indices[0]
                            text_embedding[i, idx, :] = visual_tokens[i, 0, :].to(dtype=text_embedding.dtype)
        
        vector_embedding = torch.cat([pool2.to(input_dtype), embs], dim=1).to(input_dtype)
        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):
        global DINO_EMBEDDING_DIM
        DINO_EMBEDDING_DIM = getattr(args, "dino_embedding_dim", 1024)
        
        tokenizer = self.load_tokenizer(args)
        trigger_token = getattr(args, "trigger_token", "[obj_0]")
        for tok in tokenizer:
            if trigger_token not in tok.get_vocab():
                tok.add_tokens([trigger_token])
        self.obj_0_token_id = tokenizer[1].convert_tokens_to_ids(trigger_token)
        logger.info(f"Trigger token: {trigger_token}, ID: {self.obj_0_token_id}")

        # Patch DataLoader and Accelerator to capture objects
        original_dataloader_init = torch.utils.data.DataLoader.__init__
        def patched_dataloader_init(dataloader_self, dataset, *dl_args, **dl_kwargs):
            if hasattr(dataset, "datasets") and isinstance(dataset.datasets[0], (train_util.DreamBoothDataset, train_util.FineTuningDataset)):
                 # Inject args into dataset instances so workers have access to it after pickling
                 for ds in dataset.datasets:
                     ds.v_drop_args = args

                 if "collate_fn" in dl_kwargs and hasattr(dl_kwargs["collate_fn"], "current_step"):
                     self.current_step_val = dl_kwargs["collate_fn"].current_step
            return original_dataloader_init(dataloader_self, dataset, *dl_args, **dl_kwargs)
        
        from accelerate import Accelerator
        original_prepare = Accelerator.prepare
        def patched_prepare(acc_self, *prep_args):
            results = original_prepare(acc_self, *prep_args)
            
            # Find the network among prepared objects
            prepared_objs = results if isinstance(results, tuple) else [results]
            for obj in prepared_objs:
                # We check if it looks like a network (has save_weights or apply_to)
                if hasattr(obj, "save_weights") or hasattr(obj, "apply_to"):
                    self.network_captured = obj
            return results

        torch.utils.data.DataLoader.__init__ = patched_dataloader_init
        Accelerator.prepare = patched_prepare
        try:
            super().train(args)
        finally:
            torch.utils.data.DataLoader.__init__ = original_dataloader_init
            Accelerator.prepare = original_prepare

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--trigger_token", type=str, default="[obj_0]", help="Trigger token to replace with visual embedding")
    parser.add_argument("--visual_dropout_rate", type=float, default=0.0, help="Visual dropout rate (curriculum)")
    parser.add_argument("--masked_loss_lambda", type=float, default=5.0, help="Weight lambda for masked region: 1.0 + lambda * mask")
    parser.add_argument("--dino_embedding_dim", type=int, default=1024, help="Embedding dimension of DINOv2 model")
    parser.add_argument("--enable_effilora", action="store_true", help="Enable EffiLoRA")
    parser.add_argument("--effilora_num_experts", type=int, default=4, help="Number of experts for EffiLoRA")
    parser.add_argument("--enable_slao", action="store_true", help="Enable SLAO (Merge before Forget) initialization")
    parser.add_argument("--slao_path", type=str, default=None, help="Path to previous task LoRA for SLAO")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.network_args is None: args.network_args = []

    if args.enable_effilora:
        args.network_args.append(f"enable_effilora=True")
        args.network_args.append(f"num_experts={args.effilora_num_experts}")

    if args.enable_slao:
        args.network_args.append(f"enable_slao=True")
        if args.slao_path:
            args.network_args.append(f"slao_path={args.slao_path}")

    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlVDropTrainer()
    trainer.train(args)
