import argparse
import math
import os
import random
import time
import glob
import toml

from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from library import deepspeed_utils, sdxl_model_util, sdxl_train_util, train_util, config_util
from library.utils import setup_logging, add_logging_arguments

import networks.dsc as dsc_network

class TextDistillationDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_dir, tokenizers, resolution=(1024, 1024)):
        self.prompts = []
        files = glob.glob(os.path.join(prompt_dir, "*.txt"))
        print(f"Found {len(files)} prompt files in {prompt_dir}")
        for f_path in files:
            with open(f_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                self.prompts.extend(lines)
        print(f"Total {len(self.prompts)} prompts loaded.")
        
        self.tokenizers = tokenizers
        self.resolution = resolution
        
    def __len__(self):
        # データセットの長さを定義（エポックあたりのステップ数に影響）
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]
        
        # Tokenize (SDXL specific)
        # Tokenizer 1
        tokens1 = self.tokenizers[0](prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        # Tokenizer 2
        tokens2 = self.tokenizers[1](prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        
        return {
            "input_ids": tokens1.input_ids[0],
            "input_ids2": tokens2.input_ids[0],
        }

def get_sdxl_text_embeddings(text_encoder1, text_encoder2, input_ids1, input_ids2):
    # Encoder 1
    with torch.no_grad():
        enc_out1 = text_encoder1(input_ids1, output_hidden_states=True)
        # SDXL uses penultimate layer
        hidden_states1 = enc_out1.hidden_states[11] 
        
        # Encoder 2
        enc_out2 = text_encoder2(input_ids2, output_hidden_states=True)
        hidden_states2 = enc_out2.hidden_states[-2] 
        pool2 = enc_out2.text_embeds
        
    return torch.cat([hidden_states1, hidden_states2], dim=2), pool2

def train(args):
    train_util.verify_training_args(args)
    # prompt_dirがあればdataset_argsの必須チェックを緩和したいが、
    # 既存関数を使うため、エラーが出たら引数調整が必要かもしれない。
    train_util.prepare_dataset_args(args, True) 
    setup_logging(args, reset=True)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    # 1. Setup Tokenizer
    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # 2. Setup Dataset
    if args.prompt_dir:
        print(f"Text Distillation Mode: Loading prompts from {args.prompt_dir}")
        dataset = TextDistillationDataset(args.prompt_dir, [tokenizer1, tokenizer2], resolution=(args.resolution[0], args.resolution[1]))
        # ダミーの設定
        args.max_data_loader_n_workers = 0 # シンプルにするため
    else:
        # 既存の画像データセットロード処理
        if args.dataset_class is None:
            blueprint_generator = sdxl_train_util.BlueprintGenerator(sdxl_train_util.SdxlDatasetConfiguration)
            if args.dataset_config is not None:
                print(f"Load dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    print("ignore following options because config file is used: " + ", ".join(ignored))
                config = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
                dataset = config_util.generate_dataset_by_config(config)
            else:
                print("Preparing dataset")
                dataset_config = sdxl_train_util.SdxlDatasetConfiguration()
                blueprint = blueprint_generator.generate(user_config=None, args=args, tokenizer=[tokenizer1, tokenizer2])
                dataset = config_util.generate_dataset_by_config(blueprint)
        else:
            dataset = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    # 3. Accelerator
    accelerator = train_util.prepare_accelerator(args)
    
    # 4. Load Models (Base / Teacher)
    print("Loading SDXL model...")
    text_encoder1, text_encoder2, vae, unet, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.pretrained_model_name_or_path, "cpu", args.vae_path
    )
    
    # Freeze core models
    unet.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    vae.requires_grad_(False)

    # Move to device
    unet.to(accelerator.device, dtype=train_util.get_weight_dtype(args.save_precision))
    text_encoder1.to(accelerator.device)
    text_encoder2.to(accelerator.device)
    vae.to(accelerator.device, dtype=train_util.get_weight_dtype(args.save_precision))

    # 5. Create DSC Network
    print(f"Creating DSC Network: basis={args.num_basis}, active={args.active_basis}")
    network = dsc_network.create_network(
        1.0, args.network_dim, args.network_alpha, vae, [text_encoder1, text_encoder2], unet, 
        num_basis=args.num_basis, active_basis=args.active_basis
    )
    
    # ★ SVD Initialization
    network.init_from_teacher(unet)
    
    network.apply_to(text_encoder1, text_encoder2, unet)
    network.prepare_grad_etc(text_encoder1, text_encoder2, unet)
    network.to(accelerator.device)

    # 6. Optimizer
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
    
    # 7. Dataloader & Steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(len(dataset) / args.gradient_accumulation_steps / args.train_batch_size)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.max_data_loader_n_workers)
    
    if args.save_precision == "fp16":
        unwrap_dtype = torch.float16
    elif args.save_precision == "bf16":
        unwrap_dtype = torch.bfloat16
    else:
        unwrap_dtype = torch.float32

    # Accelerate prepare
    network, optimizer, train_dataloader = accelerator.prepare(network, optimizer, train_dataloader)

    # 8. Training Loop
    print("Start DSC Distillation Training...")
    
    progress_bar = tqdm(range(args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)
    global_step = 0
    
    loss_list = []
    
    for epoch in range(args.max_train_epochs):
        network.train()
        for batch in train_dataloader:
            with accelerator.accumulate(network):
                # --- Prepare Inputs ---
                
                # 1. Latents
                if "latents" in batch:
                    # 画像ありDatasetの場合
                    latents = batch["latents"].to(accelerator.device).to(unwrap_dtype)
                else:
                    # Text Distillationの場合 -> Generate Noise Latents
                    # SDXL Latent Shape: (B, 4, H/8, W/8)
                    # resolution is tuple (W, H) in args usually? verify args.resolution format
                    # args.resolution is (W, H) usually.
                    if args.prompt_dir:
                        h, w = args.resolution
                    else:
                         # fallback
                        h, w = (1024, 1024)
                    
                    latents = torch.randn(
                        (args.train_batch_size, 4, h // 8, w // 8), 
                        device=accelerator.device, 
                        dtype=unwrap_dtype
                    )

                # 2. Text Embeddings
                if "text_encoder_outputs1_2" in batch:
                    # Cached
                    text_embeds = batch["text_encoder_outputs1_2"].to(accelerator.device).to(unwrap_dtype)
                    pool = batch["text_encoder_pool2"].to(accelerator.device).to(unwrap_dtype)
                else:
                    # On-the-fly Encoding
                    input_ids1 = batch["input_ids"].to(accelerator.device)
                    input_ids2 = batch["input_ids2"].to(accelerator.device)
                    text_embeds, pool = get_sdxl_text_embeddings(text_encoder1, text_encoder2, input_ids1, input_ids2)
                    text_embeds = text_embeds.to(unwrap_dtype)
                    pool = pool.to(unwrap_dtype)

                # 3. Time IDs (for SDXL)
                if "time_ids" in batch:
                    time_ids = batch["time_ids"].to(accelerator.device).to(unwrap_dtype)
                else:
                    # Construct Time IDs
                    def get_time_ids(h, w):
                        original_size = (h, w)
                        target_size = (h, w)
                        crop_coords = (0, 0)
                        return torch.tensor([original_size + crop_coords + target_size], device=accelerator.device)
                    
                    # Batch分複製
                    t_ids = get_time_ids(h, w)
                    time_ids = t_ids.repeat(args.train_batch_size, 1).to(unwrap_dtype)

                # --- Distillation Step ---
                
                # Noise setup
                noise = torch.randn_like(latents)
                b_size = latents.shape[0]
                timesteps = torch.randint(0, 1000, (b_size,), device=accelerator.device).long()
                
                # Add Noise to Latents (Teacher Input)
                # Teacher/Student both take noisy input and predict noise/velocity
                noisy_latents = network.noise_scheduler.add_noise(latents, noise, timesteps)

                # 1. Teacher Forward (DSC Disabled)
                network.set_multiplier(0.0) # Disable DSC
                with torch.no_grad():
                    teacher_pred = unet(noisy_latents, timesteps, text_embeds, pool, time_ids=time_ids)
                
                # 2. Student Forward (DSC Enabled)
                network.set_multiplier(1.0) # Enable DSC
                student_pred = unet(noisy_latents, timesteps, text_embeds, pool, time_ids=time_ids)

                # 3. Task Loss (Distillation MSE)
                loss_task = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")

                # 4. DSC Aux Losses
                loss_aux, loss_frame, loss_z = network.module.get_aux_losses() if hasattr(network, "module") else network.get_aux_losses()
                
                # Hyperparams
                lambda_aux = 0.01
                lambda_frame = 0.001
                lambda_z = 1e-4
                
                loss = loss_task + (lambda_aux * loss_aux) + (lambda_frame * loss_frame) + (lambda_z * loss_z)
                
                # Backward
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logs
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            current_loss = loss.detach().item()
            loss_list.append(current_loss)
            if len(loss_list) > 100: loss_list.pop(0)
            progress_bar.set_postfix(loss=sum(loss_list)/len(loss_list), aux=loss_aux.item())
            
            if global_step >= args.max_train_steps:
                break

    # Save
    print("Saving network...")
    network.save_weights(os.path.join(args.output_dir, args.output_name + ".safetensors"), unwrap_dtype, None)

def setup_parser():
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    
    # DSC Specific Args
    parser.add_argument("--num_basis", type=int, default=1024, help="DSC basis bank size")
    parser.add_argument("--active_basis", type=int, default=16, help="DSC active basis count")
    
    # Prompt Dir for Text Distillation
    parser.add_argument("--prompt_dir", type=str, default=None, help="Directory containing .txt files for prompts (Text-only distillation)")

    return parser

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    train(args)