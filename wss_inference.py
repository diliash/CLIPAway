import argparse
import os
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionInpaintPipeline
from dataset.dataset import TestDataset
from model.clip_away import CLIPAway
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from tqdm import tqdm
from torchvision import transforms
import copy
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/inference_config.yaml")
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--mask_type", type=str, required=True)
    parser.add_argument("--export_name", type=str, required=True)
    return parser.parse_args()

def dilate_mask(mask, kernel_size, iterations):
    # Ensure mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    elif isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Ensure mask is in the correct format for cv2
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    # Convert back to torch tensor
    return torch.from_numpy(dilated_mask).float().unsqueeze(0) / 255.0

def generate_focused_embeddings_grid(image, mask, fg_focused, bg_focused, projected, inpainted):
    images = [image, mask, fg_focused, bg_focused, projected, inpainted]
    row_image = Image.new('RGB', (image.width * len(images), image.height + 30))
    for i, img in enumerate(images):
        row_image.paste(img, (image.width * i, 30))

    draw = ImageDraw.Draw(row_image)
    font_path = os.path.join(os.path.dirname(__file__), "assets/OpenSans-Regular.ttf")
    font = ImageFont.truetype(font_path, 20)
    labels = ["Original Image", "Mask", "Unconditional Foreground Focused Generation",
            "Unconditional Background Focused Generation", "Unconditional CLIPAway Generation", 
            "Inpainted with CLIPAway"]

    for i, label in enumerate(labels):
        draw.text((image.width * i, 0), label, font=font, fill=(255, 255, 255))
    
    return row_image


def main(config, args):
    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        config.sd_model_key, safety_checker=None, torch_dtype=torch.float32
    ).to(device)

    clipaway = CLIPAway(
        sd_pipe=sd_pipeline, 
        image_encoder_path=config.image_encoder_path,
        ip_ckpt=config.ip_adapter_ckpt_path, 
        alpha_clip_path=config.alpha_clip_ckpt_pth, 
        config=config, 
        alpha_clip_id=config.alpha_clip_id, 
        device=device, 
        num_tokens=4
    )
    
    latents = torch.randn((1,4,64,64), generator=torch.Generator().manual_seed(config.seed)).to(device)

    scene_dirs = glob(f"{args.exp_path}/scene*")

    for scene_dir in tqdm(scene_dirs):
        print(f"Processing {scene_dir}")
        scene_id = scene_dir.split("/")[-1]
        image = Image.open(f"{args.gt_path}/{scene_id}/scene.png").convert("RGB")
        # Copy and store alpha channel
        cv2_image = cv2.imread(f"{args.gt_path}/{scene_id}/scene.png", cv2.IMREAD_UNCHANGED)
        alpha_channel = cv2_image[:, :, 3]
        
        w, h = image.size
        image = image.resize((512, 512))
        mask = Image.open(f"{args.exp_path}/{scene_id}/{args.mask_type}.png").convert("L").resize((512, 512))
        
        mask_tensor = transforms.ToTensor()(mask)
        dilated_mask = dilate_mask(mask_tensor, kernel_size=5, iterations=5)
        
        image_pil = [image]
        mask_pil = [transforms.ToPILImage()(dilated_mask.squeeze())]

        final_image = clipaway.generate(
            prompt=[""], scale=config.scale, seed=config.seed,
            pil_image=image_pil, alpha=mask_pil, strength=config.strength, latents=latents
        )[0]
        final_image = final_image.resize((w, h))
        
        if config.display_focused_embeds:
            full_mask = Image.new('L', (mask_pil[0].width, mask_pil[0].height), 255)
            projected_embeds, fg_embeds, bg_embeds, uncond_image_prompt_embeds = clipaway.get_focused_embeddings(
                image_pil, mask_pil, use_projection_block=True
            )
            
            fg_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=fg_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]

            bg_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=bg_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]

            proj_image = clipaway.generate(
                prompt=[""], image_prompt_embeds=projected_embeds, uncond_image_prompt_embeds=uncond_image_prompt_embeds,
                scale=config.scale, seed=config.seed, pil_image=image_pil,
                alpha=full_mask, strength=config.strength, latents=latents
            )[0]
            
            final_image = generate_focused_embeddings_grid(
                image_pil[0], mask_pil[0], fg_image, bg_image, proj_image, final_image
            )
        # Add alpha channel back
        final_image = np.array(final_image)
        final_image = np.dstack((final_image, alpha_channel))
        final_image = Image.fromarray(final_image)
        final_image.save(f"{scene_dir}/{args.export_name}.png")


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(config.save_path_prefix, exist_ok=True)
    main(config, args)
