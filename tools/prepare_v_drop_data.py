import argparse
import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
import warnings

# Suppress parameter renaming warnings from torch
warnings.filterwarnings("ignore", message=".*renamed internally to.*")

# Try importing GroundingDINO
try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

def setup_args():
    parser = argparse.ArgumentParser(description="Prepare data for V-Drop LoRA training (SAM2 mask & DINOv2 embedding)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--sam2_checkpoint", type=str, default="sam2_hiera_large.pt", help="Path to SAM2 checkpoint file")
    parser.add_argument("--sam2_config", type=str, default="sam2_hiera_l.yaml", help="Path to SAM2 config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    # DINOv2
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitl14", help="DINOv2 model name (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DINOv2 extraction")
    
    # Prompting modes
    parser.add_argument("--use_center_point", action="store_true", help="Use center point prompting instead of automatic mask generation.")
    
    # Text Prompting (Grounding DINO)
    parser.add_argument("--text_prompt", type=str, default=None, help="Text prompt for Grounding DINO to detect objects (e.g., 'a cat', 'person'). If set, uses Grounded-SAM2.")
    parser.add_argument("--grounding_dino_config", type=str, default="GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config")
    parser.add_argument("--grounding_dino_checkpoint", type=str, default="groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box threshold for Grounding DINO")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for Grounding DINO")

    return parser.parse_args()

def load_grounding_dino(config_path, checkpoint_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image, None)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    return boxes_filt, logits_filt.max(dim=1)[0]

def main():
    args = setup_args()
    
    # --- Collect Images ---
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    # Filter out existing mask images
    image_files = [f for f in image_files if not os.path.splitext(f)[0].lower().endswith("_mask")]
    
    print(f"Found {len(image_files)} images in {args.image_dir}.")
    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    # --- Phase 1: Mask Generation ---
    print("\n=== Phase 1: Generating Masks ===")
    
    pending_masks = [f for f in image_files if not os.path.exists(os.path.splitext(f)[0] + "_mask.png")]
    
    if pending_masks:
        # Determine mode
        mode = "auto"
        if args.text_prompt:
            mode = "text"
            if not GROUNDING_DINO_AVAILABLE:
                print("Error: GroundingDINO is not installed but --text_prompt is set.")
                print("Please install it: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
                return
            if not os.path.exists(args.grounding_dino_config) or not os.path.exists(args.grounding_dino_checkpoint):
                print(f"Error: Grounding DINO config or checkpoint not found.")
                return
        elif args.use_center_point:
            mode = "center"
        
        print(f"Mode: {mode}")
        print(f"Loading SAM2... (Processing {len(pending_masks)} images)")
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("Error: SAM2 is not installed.")
            return

        try:
            # Load SAM2
            print(f"Loading SAM2 model from {args.sam2_checkpoint}...")
            sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
            print("SAM2 loaded successfully.")
            
            mask_generator = None
            mask_predictor = None
            grounding_model = None

            if mode == "auto":
                print("Initializing Automatic Mask Generator...")
                mask_generator = SAM2AutomaticMaskGenerator(
                    sam2_model, points_per_batch=32, points_per_side=32
                )
            else:
                print("Initializing SAM2 Image Predictor...")
                mask_predictor = SAM2ImagePredictor(sam2_model)
                if mode == "text":
                    print(f"Loading Grounding DINO (prompt: '{args.text_prompt}')...")
                    grounding_model = load_grounding_dino(args.grounding_dino_config, args.grounding_dino_checkpoint, args.device)
                    print("Grounding DINO loaded successfully.")

            for img_path in tqdm(pending_masks, desc="Mask Generation"):
                mask_save_path = os.path.splitext(img_path)[0] + "_mask.png"
                try:
                    image_pil = Image.open(img_path).convert("RGB")
                    image_np = np.array(image_pil)
                    h, w = image_np.shape[:2]
                    
                    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                    
                    mask_bool = None
                    
                    with torch.inference_mode(), torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=autocast_dtype):
                        
                        if mode == "text":
                            # 1. Detect boxes with Grounding DINO
                            # Grounding DINO's CUDA kernel doesn't support bf16, so we disable autocast here
                            with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", enabled=False):
                                boxes_filt, scores = get_grounding_output(
                                    grounding_model, image_pil, args.text_prompt, args.box_threshold, args.text_threshold, args.device
                                )
                            
                            if boxes_filt.shape[0] == 0:
                                print(f"No object found for '{args.text_prompt}' in {img_path}")
                            else:
                                # Convert boxes from cxcywh to xyxy and scale to image size
                                boxes_filt = boxes_filt * torch.Tensor([w, h, w, h]).to(boxes_filt.device)
                                boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
                                boxes_filt[:, 2:] += boxes_filt[:, :2]
                                boxes_xyxy = boxes_filt.cpu().numpy()

                                mask_predictor.set_image(image_np)
                                
                                # Predict masks for all boxes
                                masks, _, _ = mask_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=boxes_xyxy,
                                    multimask_output=False,
                                )
                                # masks shape check
                                if masks.ndim == 4:
                                    masks = masks[:, 0, :, :]
                                
                                # Choose the largest mask among detected objects (or highest score box?)
                                # Here we choose the box with highest score from Grounding DINO
                                best_box_idx = torch.argmax(scores).item()
                                mask_bool = masks[best_box_idx].astype(bool)

                        elif mode == "center":
                            mask_predictor.set_image(image_np)
                            input_point = np.array([[w // 2, h // 2]])
                            input_label = np.array([1])
                            
                            masks, scores, _ = mask_predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                            )
                            best_idx = np.argmax(scores)
                            mask_bool = masks[best_idx].astype(bool)

                        else: # auto
                            masks = mask_generator.generate(image_np)
                            if len(masks) > 0:
                                sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                                best_mask_data = sorted_masks[0]
                                img_area = image_np.shape[0] * image_np.shape[1]
                                if best_mask_data['area'] / img_area > 0.95 and len(sorted_masks) > 1:
                                     best_mask_data = sorted_masks[1]
                                mask_bool = best_mask_data['segmentation']

                    if mask_bool is not None:
                        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8))
                        mask_img.save(mask_save_path)
                    
                    # Cleanup per image
                    del mask_bool, image_np, image_pil
                    if mode == "text" and 'boxes_filt' in locals(): del boxes_filt, scores
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    torch.cuda.empty_cache()
            
            # Unload models
            del mask_generator, mask_predictor, grounding_model, sam2_model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed during Mask Generation phase: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("All masks already exist.")


    # --- Phase 2: DINOv2 Embedding ---
    print("\n=== Phase 2: Extracting Embeddings with DINOv2 (Batched) ===")
    
    pending_dino = [f for f in image_files if not os.path.exists(os.path.splitext(f)[0] + ".npy")]
            
    if pending_dino:
        print(f"Loading DINOv2 ({args.dino_model_name})...")
        try:
            dino_model = torch.hub.load('facebookresearch/dinov2', args.dino_model_name).to(args.device)
            dino_model.eval()
            
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            # Batch processing for DINOv2
            # batch_size=8 is safe
            for i in tqdm(range(0, len(pending_dino), args.batch_size), desc="DINOv2 Batched"):
                batch_paths = pending_dino[i : i + args.batch_size]
                batch_tensors = []
                valid_indices = []

                for idx, img_path in enumerate(batch_paths):
                    mask_save_path = os.path.splitext(img_path)[0] + "_mask.png"
                    if not os.path.exists(mask_save_path):
                        continue
                    
                    try:
                        image_pil = Image.open(img_path).convert("RGB")
                        mask_pil = Image.open(mask_save_path).convert("L")
                        mask_np = np.array(mask_pil) > 128
                        
                        image_np = np.array(image_pil)
                        if mask_np.shape != image_np.shape[:2]:
                             mask_pil = mask_pil.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
                             mask_np = np.array(mask_pil) > 128
                        
                        image_np[~mask_np] = 0
                        tensor = transform(Image.fromarray(image_np))
                        batch_tensors.append(tensor)
                        valid_indices.append(idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

                if not batch_tensors:
                    continue

                input_batch = torch.stack(batch_tensors).to(args.device)
                with torch.inference_mode():
                    embeddings = dino_model(input_batch).cpu().numpy()

                for idx, embedding in zip(valid_indices, embeddings):
                    npy_save_path = os.path.splitext(batch_paths[idx])[0] + ".npy"
                    np.save(npy_save_path, embedding)
                
                del input_batch, embeddings, batch_tensors
                torch.cuda.empty_cache()
            
            del dino_model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed during DINOv2 phase: {e}")
            return
    else:
        print("All embeddings already exist.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
