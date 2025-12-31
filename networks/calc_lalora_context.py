import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import logging

# network.pyをインポート
import network as universal_network_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalImageDataset(Dataset):
    def __init__(self, img_dir, size=512):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        self.size = size
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image}

def load_model(path, device, dtype):
    """ファイルパスに応じてSD1.5またはSDXLを読み込む"""
    is_single_file = os.path.isfile(path)
    use_sdxl = "sdxl" in path.lower() or "xl" in path.lower()
    
    logger.info(f"Loading {'SDXL' if use_sdxl else 'SD1.5'} model from {path}...")
    
    # ロードは安全のためfloat16で (fp8指定時もロード自体はfp16で行うのが無難)
    # 最終的にUNetだけfp8にする
    load_dtype = dtype if dtype != torch.float8_e4m3fn else torch.float16

    if use_sdxl:
        if is_single_file:
            pipeline = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=load_dtype)
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=load_dtype)
    else:
        if is_single_file:
            pipeline = StableDiffusionPipeline.from_single_file(path, torch_dtype=load_dtype)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(path, torch_dtype=load_dtype)

    return pipeline, use_sdxl

def main():
    parser = argparse.ArgumentParser(description="Calculate LaLoRA Context (Precision & Mean) - Optimized")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--basis_path", type=str, required=True)
    parser.add_argument("--source_data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision setting")
    parser.add_argument("--fp8_base", action="store_true", help="Cast UNet to fp8 (float8_e4m3fn). Text Encoder remains in mixed_precision dtype.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--xformers", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    # 高速化設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 混合精度の設定
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 1. Load Models
    pipeline, is_sdxl = load_model(args.pretrained_model_name_or_path, args.device, weight_dtype)
    
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler

    text_encoder_2 = None
    if is_sdxl:
        text_encoder_2 = pipeline.text_encoder_2

    # --- Feature: xformers ---
    if args.xformers:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled.")
        except ImportError:
            logger.warning("xformers not found. Skipping.")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    # --- Feature: Gradient Checkpointing ---
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient Checkpointing enabled for UNet.")

    # --- Feature: FP8 Base ---
    if args.fp8_base:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("Your PyTorch version does not support float8_e4m3fn. Please upgrade or disable --fp8_base.")
        
        logger.info("Casting UNet to float8_e4m3fn (Text Encoders remain in original dtype)...")
        unet.to(dtype=torch.float8_e4m3fn)
        # Text Encoderはキャストしない (互換性維持)

    # モデル配置
    text_encoder.to(args.device)
    unet.to(args.device)
    if text_encoder_2: text_encoder_2.to(args.device)
    
    # VAEはfp8非推奨。ロード時のdtype (fp16/bf16/fp32) を維持
    vae.to(args.device, dtype=weight_dtype) 

    # 凍結
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    if is_sdxl: text_encoder_2.requires_grad_(False)

    # 2. Initialize Universal Network
    logger.info(f"Initializing Universal Network...")
    te_arg = [text_encoder, text_encoder_2] if is_sdxl else text_encoder
    
    network = universal_network_module.create_network(
        multiplier=args.multiplier,
        network_dim=None, network_alpha=None, vae=vae, text_encoder=te_arg, unet=unet,
        basis_path=args.basis_path, lalora_lambda=0.0
    )
    
    network.apply_to(te_arg, unet, True, True)
    network.prepare_grad_etc(te_arg, unet)
    network.to(args.device)

    # 3. Pre-compute Text Embeddings (Cache)
    logger.info("Pre-computing text embeddings...")
    
    # Autocastの設定
    # fp8_baseのときは推論精度確保のためbf16推奨だが、mixed_precision設定に従う
    autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    with torch.no_grad():
        # torch.cuda.amp.autocast -> torch.amp.autocast
        with torch.amp.autocast('cuda', enabled=True, dtype=autocast_dtype): 
            if is_sdxl:
                prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                    prompt="", device=args.device, do_classifier_free_guidance=False
                )
                original_size = (args.resolution, args.resolution)
                target_size = (args.resolution, args.resolution)
                crops_coords_top_left = (0, 0)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], device=args.device, dtype=prompt_embeds.dtype)
                
                cached_encoder_hidden_states = prompt_embeds
                cached_added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            else:
                input_ids = pipeline.tokenizer(
                    "", padding="max_length", truncation=True, 
                    max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
                ).input_ids.to(args.device)
                cached_encoder_hidden_states = text_encoder(input_ids)[0]
                cached_added_cond_kwargs = {}

    # 4. Dataset & Dataloader
    dataset = LocalImageDataset(args.source_data_dir, size=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 5. Calculate Curvature
    logger.info("Starting curvature calculation...")
    
    precision_accum = {}
    for name, module in network.modules_dict.items():
        precision_accum[name] = torch.zeros_like(module.alpha, device=args.device, dtype=torch.float32)

    network.train()
    step_count = 0
    progress_bar = tqdm(total=args.max_steps)
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    while step_count < args.max_steps:
        for batch in dataloader:
            if step_count >= args.max_steps:
                break
            
            current_batch_size = batch["pixel_values"].shape[0]
            
            with torch.amp.autocast('cuda', enabled=(args.mixed_precision != "no"), dtype=autocast_dtype):
                # VAE Encode (VAEはfp8ではないのでweight_dtype)
                latents = vae.encode(batch["pixel_values"].to(args.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Expand cached text embeddings
                batch_encoder_hidden_states = cached_encoder_hidden_states.repeat(current_batch_size, 1, 1)
                
                batch_added_cond_kwargs = {}
                if is_sdxl:
                    batch_added_cond_kwargs = {
                        "text_embeds": cached_added_cond_kwargs["text_embeds"].repeat(current_batch_size, 1),
                        "time_ids": cached_added_cond_kwargs["time_ids"].repeat(current_batch_size, 1)
                    }

                # Predict noise
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    batch_encoder_hidden_states, 
                    added_cond_kwargs=batch_added_cond_kwargs
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Backward
            scaler.scale(loss).backward()

            # Accumulate squared gradients
            with torch.no_grad():
                for name, module in network.modules_dict.items():
                    if module.alpha.grad is not None:
                        grads_squared = module.alpha.grad.pow(2).to(torch.float32)
                        precision_accum[name] += grads_squared
                        module.alpha.grad = None
            
            step_count += 1
            progress_bar.update(1)

    # 6. Save
    logger.info("Saving context file...")
    state_dict = {}
    for name, module in network.modules_dict.items():
        state_dict[f"{name}.alpha_mean"] = module.alpha.detach().cpu().to(torch.float32)
        prec_val = (precision_accum[name] / step_count).detach().cpu().to(torch.float32)
        state_dict[f"{name}.alpha_precision"] = prec_val

    save_file(state_dict, args.output_path)
    logger.info(f"Saved LaLoRA context to {args.output_path}")

if __name__ == "__main__":
    main()