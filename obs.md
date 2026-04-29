## 2026.4.29
### inference效果差问题
#### inference-3t修改
```
import argparse
import logging
import math
import os
from typing import Optional, Tuple
# os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
import random
import pandas as pd
import shutil
from pathlib import Path
from wavelet_fix import *
import numpy as np
import PIL
import safetensors
from tqdm.auto import trange, tqdm
from PISA_attn_processor import AttnProcessorDistReciprocal
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from huggingface_hub import create_repo, upload_folder
from custom_diffusers.pipeline_sdxl import TiledStableDiffusionXLPipeline
from custom_diffusers.attention_processor import AttnProcessor2_0
from dataset import TestDataset
import piq
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import re
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from classical_renderer.scatter import ModuleRenderScatter  # circular aperture
# from classical_renderer.scatter_ex import ModuleRenderScatterEX  # adjustable aperture shape

# 与 train_pisa_per_step.py 对齐：同一前向内每个 denoise 子步单独计算 pisa_strength
def compute_pisa_strength_denoise_step(
    sub_step_index: int,
    num_denoise_steps: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    if num_denoise_steps <= 1:
        return float(end_ratio)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_ratio + (end_ratio - start_ratio) * progress)


def parse_step_update_scales(raw_scales: str):
    if raw_scales is None:
        return None
    raw_scales = raw_scales.strip()
    if raw_scales == "":
        return None
    try:
        values = [float(v.strip()) for v in raw_scales.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError("--step_update_scales must be a comma-separated float list.") from exc
    if len(values) == 0:
        return None
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError("--step_update_scales values must be in [0, 1].")
    return values


def compute_step_update_scale(
    sub_step_index: int,
    num_denoise_steps: int,
    explicit_scales,
    start_scale: float,
    end_scale: float,
) -> float:
    if explicit_scales is not None:
        return float(explicit_scales[sub_step_index])
    if num_denoise_steps <= 1:
        return float(end_scale)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_scale + (end_scale - start_scale) * progress)


# 与 train.py 中 gamma_correction 一致，保证 VAE 输入域与训练对齐
def gamma_correction(image: torch.Tensor, gamma: float = 2.2, eps=1e-7, upper_clip=None):
    if gamma == 1:
        return image
    base = image.min().detach()
    span = (image.max() - base).detach()
    image_norm = torch.clamp((image - base) / span, eps, upper_clip).pow(gamma)
    return image_norm * span + base

def swap_words(s: str, x: str, y: str):
    return s.replace(x, chr(0)).replace(y, x).replace(chr(0), y)


def _vae_output_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert train-space image tensor [-1, 1] to uint8 RGB."""
    if image.dim() == 4:
        image = image[0]
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = (image * 255).round().byte()
    return image.permute(1, 2, 0).cpu().numpy()


def _decode_latents_to_pil(
    pipeline,
    latents: torch.Tensor,
    resize_to: Optional[Tuple[int, int]],
    gamma: float = 1.0,
) -> Image.Image:
    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    image = gamma_correction(image, 1.0 / gamma)
    image = torch.clamp(image, -1, 1)
    image_np = _vae_output_to_uint8(image)
    pil = Image.fromarray(image_np)
    if resize_to is not None:
        rw, rh = resize_to
        pil = pil.resize((int(rw), int(rh)), resample=Image.Resampling.LANCZOS)
    return pil


def _save_pil_path(path: str, pil_img: Image.Image, ext: str) -> None:
    if ext.lower() in ("jpg", "jpeg"):
        pil_img.save(path, quality=95, subsampling=0)
    else:
        pil_img.save(path)


logger = get_logger(__name__)

# 修改attention模块
def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

import time

@torch.inference_mode()
def log_validation(
    pipeline,
    vae,
    seed,
    train_T_list,
    accelerator,
    validation_prompt="an excellent photo with a large aperture",
    aif_image=None,
    disp_coc=None,
    gamma: float = 1.0,
    pisa_ratio_start: float = 1.0,
    pisa_ratio_end: float = 0.0,
    step_update_scales=None,
    step_k_start: float = 1.0,
    step_k_end: float = 1.0,
    step_save_dir: Optional[str] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    save_initial_encode: bool = False,
    step_image_ext: str = "png",
):
    device = accelerator.device

    dtype = pipeline.unet.dtype  # latent dtype for unet/vae

    # --- 准备输入（严格对齐 train-3t.py）---
    aif_image = gamma_correction(aif_image.to(device=device, dtype=torch.float32), gamma).to(dtype=dtype)
    latents = vae.encode(aif_image).latent_dist.mode()
    latents = latents * pipeline.vae.config.scaling_factor
    latents = latents.to(device=device, dtype=dtype)

    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=validation_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # --- 构造 time_ids（与 train 结构一致，保持 float32） ---
    _, _, h, w = latents.shape
    height = h * 8
    width = w * 8
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        device=device,
        dtype=torch.float32,
    ).repeat(latents.shape[0], 1)

    # --- 其它条件输入 ---
    pooled = pooled_prompt_embeds.to(device=device, dtype=torch.float32)
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float32)
    disp_coc = disp_coc.to(device=device, dtype=torch.float32)

    # --- train 仅使用 noise_scheduler.alphas_cumprod ---
    scheduler = pipeline.scheduler
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    if step_save_dir is not None:
        os.makedirs(step_save_dir, exist_ok=True)
        if save_initial_encode:
            pil0 = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            _save_pil_path(
                os.path.join(step_save_dir, f"step000_init_encode.{step_image_ext}"),
                pil0,
                step_image_ext,
            )

    # --- 迭代 denoise（严格对齐 train-3t.py 前向）---
    n_steps = len(train_T_list)
    for sub_i, t in enumerate(train_T_list):
        pisa_strength = compute_pisa_strength_denoise_step(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            start_ratio=pisa_ratio_start,
            end_ratio=pisa_ratio_end,
        )
        t_tensor = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)

        noise_pred = pipeline.unet(
            latents,
            t_tensor,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled,
                "time_ids": time_ids,
            },
            cross_attention_kwargs={"disp_coc": disp_coc, "pisa_strength": float(pisa_strength)},
        ).sample

        alpha_prod_t = scheduler.alphas_cumprod[t_tensor][:, None, None, None]
        beta_prod_t = 1.0 - alpha_prod_t
        latents_f = latents.to(torch.float32)
        noise_pred_f = noise_pred.to(torch.float32)
        latents_full = (latents_f - beta_prod_t.sqrt() * noise_pred_f) / alpha_prod_t.sqrt()
        step_k = compute_step_update_scale(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            explicit_scales=step_update_scales,
            start_scale=step_k_start,
            end_scale=step_k_end,
        )
        latents = (latents_f + step_k * (latents_full - latents_f)).to(dtype=dtype)

        if step_save_dir is not None:
            pil = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            name = f"step{sub_i + 1:03d}_t{t:04d}_pisa{float(pisa_strength):.2f}.{step_image_ext}"
            _save_pil_path(os.path.join(step_save_dir, name), pil, step_image_ext)
            print(f"saved {name}")



    # --- 解码输出（严格对齐 train：decode -> gamma^-1 -> clamp[-1,1]）---
    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False
    )[0]
    image = gamma_correction(image, 1.0 / gamma)
    image = torch.clamp(image, -1, 1)

    return (image, None)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference (PISA 权重在 train_T_list 各子步上线性变化，对齐 train_pisa_per_step.py)。",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./models/sdxl-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        help="Variant of BokehDiff model. Currently we only support model size of `small`.",
        choices=["small"]
    )
    parser.add_argument(
        "--test_data_dir", type=str, default="./test_data/input/*.jpg", help="A folder containing the testing data."
    )
    parser.add_argument(
        "--train_T_list",
        type=int,
        nargs="+",
        default=[499, 300, 100],
        help="Denoising timesteps, e.g. --train_T_list 499 300 100",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bokehdiff_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--K", type=float, default=20, help="Param of the aperture. Larger K -> more blur.")
    parser.add_argument("--upsample", type=float, default=1, help="Perform upsampling on the image before rendering in latent space.")
    parser.add_argument(
        "--data_id",
        type=str,
        default="demo",
        help="The folder name of the current inference run",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--organization",
        type=str,
        default="EBB",
        help=("File organization of testing dataset. Should be 'folder', 'pngdepth', or 'EBB'."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="./output/run6/checkpoint-70000",
        help=(
            "Directory containing pytorch_lora_weights.safetensors and vae.ckpt (same layout as train checkpoints)."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="VAE 编码前 gamma，与 train.py --gamma 一致（默认 1 即不变换）。",
    )
    parser.add_argument(
        "--pisa_ratio_start",
        type=float,
        default=1.0,
        help="与 train_pisa_per_step 一致：train_T_list 第一步的 PISA 比例。",
    )
    parser.add_argument(
        "--pisa_ratio_end",
        type=float,
        default=0.0,
        help="与 train_pisa_per_step 一致：train_T_list 最后一步的 PISA 比例。",
    )
    parser.add_argument(
        "--step_update_scales",
        type=str,
        default="",
        help=(
            "显式指定每个去噪子步的更新系数 k_i，逗号分隔（例如 1.0,0.75,0.5）。"
            " 若为空，则使用 step_k_start->step_k_end 线性插值。"
        ),
    )
    parser.add_argument(
        "--step_k_start",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，第一个子步的更新系数。",
    )
    parser.add_argument(
        "--step_k_end",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，最后一个子步的更新系数。",
    )
    parser.add_argument(
        "--supersampling_num",
        type=int,
        default=4,
        help="与训练 AttnProcessorDistReciprocal 一致（train 默认 4）。",
    )
    parser.add_argument(
        "--segment_num",
        type=int,
        default=5,
        help="与训练 AttnProcessorDistReciprocal 一致（train 默认 5）。",
    )
    parser.add_argument(
        "--use_dataset_k",
        action="store_true",
        help="若 batch 含 K 且与训练时一致（如 defocus*K，K 已为 K_strength/10），用其构造 disp_coc 第二通道；默认用 --K 与 train 中手动尺度。",
    )
    parser.add_argument(
        "--no_save_intermediate",
        action="store_true",
        help="默认将每个去噪子步后的 latents 解码并保存到子目录；加此标志则只保存最终 jpg。",
    )
    parser.add_argument(
        "--save_initial_encode",
        action="store_true",
        help="保存 VAE 编码后、去噪前的图 step000_init_encode.*（需未加 --no_save_intermediate）。",
    )
    parser.add_argument(
        "--step_image_ext",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="中间步保存格式。",
    )
    
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    if not (0.0 <= args.pisa_ratio_start <= 1.0 and 0.0 <= args.pisa_ratio_end <= 1.0):
        raise ValueError("--pisa_ratio_start and --pisa_ratio_end must be in [0, 1].")
    if not (0.0 <= args.step_k_start <= 1.0 and 0.0 <= args.step_k_end <= 1.0):
        raise ValueError("--step_k_start and --step_k_end must be in [0, 1].")
    args.step_update_scales = parse_step_update_scales(args.step_update_scales)
    if args.step_update_scales is not None and len(args.step_update_scales) != len(args.train_T_list):
        raise ValueError(
            f"--step_update_scales expects {len(args.train_T_list)} values for train_T_list={args.train_T_list}, "
            f"but got {len(args.step_update_scales)}."
        )
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.test_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def main():
    print("main...")
    args = parse_args()
    print("args...")
    
    import os
    dist_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
    for v in dist_vars:
        if v in os.environ:
            del os.environ[v]
            print(f">>> 已清除环境变量: {v}")

    # 初始化 Accelerator
    # 显式指定 cpu=False 和 mixed_precision，不依赖自动配置
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=False
    )
    
    print(f">>> [3] Accelerator 初始化完成！设备: {accelerator.device}")

    print("accelerator = Accelerator...")
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print("logging.basicConfig...")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    print("load tokenizer...")
    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, use_safetensors=True,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,use_safetensors=True,
    ).eval()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True,
    ).eval()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    vae.enable_tiling()
    text_encoder_1.eval()
    text_encoder_2.eval()
    logging.disable(logging.CRITICAL) # Filter out warnings

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = TiledStableDiffusionXLPipeline(
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    val_dataset = TestDataset(args.test_data_dir,
        tokenizer_1=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        organization = args.organization,
        split="inference",
        upsample=args.upsample,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True
    )

    ckpt_dir = args.resume_from_checkpoint
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise ValueError(
            f"--resume_from_checkpoint must be an existing directory with LoRA+VAE weights, got: {ckpt_dir!r}"
        )
    accelerator.print(f"Loading weights from {ckpt_dir}")
    pipeline.load_lora_weights(ckpt_dir, weight_name="pytorch_lora_weights.safetensors")
    vae_path = os.path.join(ckpt_dir, "vae.ckpt")
    try:
        vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
    except TypeError:
        vae_sd = torch.load(vae_path, map_location="cpu")
    pipeline.vae.load_state_dict(vae_sd, strict=False)


    # pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet)
    # pipeline, val_dataloader = accelerator.prepare(pipeline, val_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    pipeline = pipeline.to(accelerator.device)


    fn_recursive_attn_processor(
        "unet",
        pipeline.unet,
        AttnProcessorDistReciprocal(
            hard=1e7,
            supersampling_num=args.supersampling_num,
            segment_num=args.segment_num,
            train=False,
        ),
    )
    output_dir = os.path.join(args.output_dir, args.data_id)
    os.makedirs(output_dir, exist_ok=True)

    durations = []

    for batch in tqdm(val_dataloader):
        torch.cuda.empty_cache()

        blur_strength_list = [0.2, 0.4, 0.6, 0.8, 1.0]

        for blur_strength in blur_strength_list:
            defocus_strength = batch["defocus_strength"] * blur_strength
            disparity = batch["disparity"]
            # 与 train-tmp.py 一致：coc2 = defocus * (K_cli*upsample/10)
            if args.use_dataset_k and "K" in batch and batch["K"] is not None:
                k = batch["K"].to(device=accelerator.device, dtype=torch.float32)
                while k.dim() < 4:
                    k = k.unsqueeze(-1)
                coc2 = defocus_strength * k
            else:
                # train 中第二通道为 defocus_strength * K（K 在数据侧已归一到训练尺度）
                amplify = args.K / 10.0
                coc2 = defocus_strength * amplify
            h, w = coc2.shape[-2:]
            new_w, new_h = int(w / args.upsample), int(h / args.upsample)
            step_dir = None
            if not args.no_save_intermediate:
                step_dir = os.path.join(
                    output_dir,
                    f"{batch['filename'][0]}_blur{blur_strength:.2f}_pisa_step_frames",
                )

            # 推理（子步 pisa 与 train_pisa_per_step 一致；默认每步保存解码图到 step_dir）
            image_tensor, duration = log_validation(
                pipeline,
                pipeline.vae,
                args.seed,
                args.train_T_list,
                accelerator,
                validation_prompt=batch['texts'],
                aif_image=batch['pixel_values'],
                disp_coc=torch.cat([disparity, coc2], 1).to(weight_dtype),
                gamma=args.gamma,
                pisa_ratio_start=args.pisa_ratio_start,
                pisa_ratio_end=args.pisa_ratio_end,
                step_update_scales=args.step_update_scales,
                step_k_start=args.step_k_start,
                step_k_end=args.step_k_end,
                step_save_dir=step_dir,
                resize_to=(new_w, new_h) if step_dir is not None else None,
                save_initial_encode=args.save_initial_encode,
                step_image_ext=args.step_image_ext,
            )
            if duration is not None:
                durations.append(duration)

            # --- tensor -> PIL.Image（对齐 train：[-1,1] -> [0,1]）---
            image_np = _vae_output_to_uint8(image_tensor)
            image = Image.fromarray(image_np)

            # resize 并保存（new_w/new_h 已在上面与中间步图对齐）
            image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            out_path = f"{output_dir}/{batch['filename'][0]}_blur{blur_strength:.2f}.jpg"
            image.save(out_path, subsampling=0, quality=100)


    if durations:
        print(f"Average duration: {np.mean(durations):.4f} seconds")


if __name__ == "__main__":
    main()

```
#### inference-3t-strict：严格对齐train-3t
```
import argparse
import logging
import math
import os
from typing import Optional, Tuple
# os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
import random
import pandas as pd
import shutil
from pathlib import Path
from wavelet_fix import *
import numpy as np
import PIL
import safetensors
from tqdm.auto import trange, tqdm
from PISA_attn_processor import AttnProcessorDistReciprocal
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from huggingface_hub import create_repo, upload_folder
from custom_diffusers.pipeline_sdxl import TiledStableDiffusionXLPipeline
from custom_diffusers.attention_processor import AttnProcessor2_0
from dataset import TestDataset
import piq
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import re
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from classical_renderer.scatter import ModuleRenderScatter  # circular aperture
# from classical_renderer.scatter_ex import ModuleRenderScatterEX  # adjustable aperture shape

# 与 train_pisa_per_step.py 对齐：同一前向内每个 denoise 子步单独计算 pisa_strength
def compute_pisa_strength_denoise_step(
    sub_step_index: int,
    num_denoise_steps: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    if num_denoise_steps <= 1:
        return float(end_ratio)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_ratio + (end_ratio - start_ratio) * progress)


def parse_step_update_scales(raw_scales: str):
    if raw_scales is None:
        return None
    raw_scales = raw_scales.strip()
    if raw_scales == "":
        return None
    try:
        values = [float(v.strip()) for v in raw_scales.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError("--step_update_scales must be a comma-separated float list.") from exc
    if len(values) == 0:
        return None
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError("--step_update_scales values must be in [0, 1].")
    return values


def compute_step_update_scale(
    sub_step_index: int,
    num_denoise_steps: int,
    explicit_scales,
    start_scale: float,
    end_scale: float,
) -> float:
    if explicit_scales is not None:
        return float(explicit_scales[sub_step_index])
    if num_denoise_steps <= 1:
        return float(end_scale)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_scale + (end_scale - start_scale) * progress)


# 与 train.py 中 gamma_correction 一致，保证 VAE 输入域与训练对齐
def gamma_correction(image: torch.Tensor, gamma: float = 2.2, eps=1e-7, upper_clip=None):
    if gamma == 1:
        return image
    base = image.min().detach()
    span = (image.max() - base).detach()
    image_norm = torch.clamp((image - base) / span, eps, upper_clip).pow(gamma)
    return image_norm * span + base

def swap_words(s: str, x: str, y: str):
    return s.replace(x, chr(0)).replace(y, x).replace(chr(0), y)


def _vae_output_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert train-space image tensor [-1, 1] to uint8 RGB."""
    if image.dim() == 4:
        image = image[0]
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = (image * 255).round().byte()
    return image.permute(1, 2, 0).cpu().numpy()


def _decode_latents_to_pil(
    pipeline,
    latents: torch.Tensor,
    resize_to: Optional[Tuple[int, int]],
    gamma: float = 1.0,
) -> Image.Image:
    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    image = gamma_correction(image, 1.0 / gamma)
    image = torch.clamp(image, -1, 1)
    image_np = _vae_output_to_uint8(image)
    pil = Image.fromarray(image_np)
    if resize_to is not None:
        rw, rh = resize_to
        pil = pil.resize((int(rw), int(rh)), resample=Image.Resampling.LANCZOS)
    return pil


def _save_pil_path(path: str, pil_img: Image.Image, ext: str) -> None:
    if ext.lower() in ("jpg", "jpeg"):
        pil_img.save(path, quality=95, subsampling=0)
    else:
        pil_img.save(path)


logger = get_logger(__name__)

# 修改attention模块
def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

import time

@torch.inference_mode()
def log_validation(
    pipeline,
    vae,
    seed,
    train_T_list,
    accelerator,
    validation_prompt="an excellent photo with a large aperture",
    aif_image=None,
    disp_coc=None,
    gamma: float = 1.0,
    pisa_ratio_start: float = 1.0,
    pisa_ratio_end: float = 0.0,
    step_update_scales=None,
    step_k_start: float = 1.0,
    step_k_end: float = 1.0,
    step_save_dir: Optional[str] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    original_size: Optional[Tuple[int, int]] = None,
    crop_top_left: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    save_initial_encode: bool = False,
    step_image_ext: str = "png",
):
    device = accelerator.device

    dtype = pipeline.unet.dtype  # latent dtype for unet/vae

    # --- 准备输入（严格对齐 train-3t.py）---
    aif_image = gamma_correction(aif_image.to(device=device, dtype=torch.float32), gamma).to(dtype=dtype)
    latents = vae.encode(aif_image).latent_dist.mode()
    latents = latents * pipeline.vae.config.scaling_factor
    latents = latents.to(device=device, dtype=dtype)

    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=validation_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # --- 构造 time_ids（严格按 train 规则：original_size + crop_top_left + target_size）---
    if original_size is None or crop_top_left is None or target_size is None:
        raise ValueError("strict mode requires original_size/crop_top_left/target_size to build time_ids.")
    bsz = latents.shape[0]
    time_ids = torch.stack(
        [torch.tensor(original_size + crop_top_left + target_size, device=device, dtype=torch.float32) for _ in range(bsz)],
        dim=0,
    )

    # --- 其它条件输入 ---
    pooled = pooled_prompt_embeds.to(device=device, dtype=torch.float32)
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float32)
    disp_coc = disp_coc.to(device=device, dtype=torch.float32)

    # --- train 仅使用 noise_scheduler.alphas_cumprod ---
    scheduler = pipeline.scheduler
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    if step_save_dir is not None:
        os.makedirs(step_save_dir, exist_ok=True)
        if save_initial_encode:
            pil0 = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            _save_pil_path(
                os.path.join(step_save_dir, f"step000_init_encode.{step_image_ext}"),
                pil0,
                step_image_ext,
            )

    # --- 迭代 denoise（严格对齐 train-3t.py 前向）---
    n_steps = len(train_T_list)
    for sub_i, t in enumerate(train_T_list):
        pisa_strength = compute_pisa_strength_denoise_step(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            start_ratio=pisa_ratio_start,
            end_ratio=pisa_ratio_end,
        )
        t_tensor = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)

        noise_pred = pipeline.unet(
            latents,
            t_tensor,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled,
                "time_ids": time_ids,
            },
            cross_attention_kwargs={"disp_coc": disp_coc, "pisa_strength": float(pisa_strength)},
        ).sample

        alpha_prod_t = scheduler.alphas_cumprod[t_tensor][:, None, None, None]
        beta_prod_t = 1.0 - alpha_prod_t
        latents_f = latents.to(torch.float32)
        noise_pred_f = noise_pred.to(torch.float32)
        latents_full = (latents_f - beta_prod_t.sqrt() * noise_pred_f) / alpha_prod_t.sqrt()
        step_k = compute_step_update_scale(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            explicit_scales=step_update_scales,
            start_scale=step_k_start,
            end_scale=step_k_end,
        )
        latents = (latents_f + step_k * (latents_full - latents_f)).to(dtype=dtype)

        if step_save_dir is not None:
            pil = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            name = f"step{sub_i + 1:03d}_t{t:04d}_pisa{float(pisa_strength):.2f}.{step_image_ext}"
            _save_pil_path(os.path.join(step_save_dir, name), pil, step_image_ext)
            print(f"saved {name}")



    # --- 解码输出（严格对齐 train：decode -> gamma^-1 -> clamp[-1,1]）---
    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False
    )[0]
    image = gamma_correction(image, 1.0 / gamma)
    image = torch.clamp(image, -1, 1)

    return (image, None)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference (PISA 权重在 train_T_list 各子步上线性变化，对齐 train_pisa_per_step.py)。",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./models/sdxl-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        help="Variant of BokehDiff model. Currently we only support model size of `small`.",
        choices=["small"]
    )
    parser.add_argument(
        "--test_data_dir", type=str, default="./test_data/input/*.jpg", help="A folder containing the testing data."
    )
    parser.add_argument(
        "--train_T_list",
        type=int,
        nargs="+",
        default=[499, 300, 100],
        help="Denoising timesteps, e.g. --train_T_list 499 300 100",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Strict mode target_size in time_ids, aligned with train --resolution.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bokehdiff_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--K", type=float, default=20, help="Param of the aperture. Larger K -> more blur.")
    parser.add_argument("--upsample", type=float, default=1, help="Perform upsampling on the image before rendering in latent space.")
    parser.add_argument(
        "--data_id",
        type=str,
        default="demo",
        help="The folder name of the current inference run",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--organization",
        type=str,
        default="EBB",
        help=("File organization of testing dataset. Should be 'folder', 'pngdepth', or 'EBB'."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="./output/run6/checkpoint-70000",
        help=(
            "Directory containing pytorch_lora_weights.safetensors and vae.ckpt (same layout as train checkpoints)."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="VAE 编码前 gamma，与 train.py --gamma 一致（默认 1 即不变换）。",
    )
    parser.add_argument(
        "--pisa_ratio_start",
        type=float,
        default=1.0,
        help="与 train_pisa_per_step 一致：train_T_list 第一步的 PISA 比例。",
    )
    parser.add_argument(
        "--pisa_ratio_end",
        type=float,
        default=0.0,
        help="与 train_pisa_per_step 一致：train_T_list 最后一步的 PISA 比例。",
    )
    parser.add_argument(
        "--step_update_scales",
        type=str,
        default="",
        help=(
            "显式指定每个去噪子步的更新系数 k_i，逗号分隔（例如 1.0,0.75,0.5）。"
            " 若为空，则使用 step_k_start->step_k_end 线性插值。"
        ),
    )
    parser.add_argument(
        "--step_k_start",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，第一个子步的更新系数。",
    )
    parser.add_argument(
        "--step_k_end",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，最后一个子步的更新系数。",
    )
    parser.add_argument(
        "--supersampling_num",
        type=int,
        default=4,
        help="与训练 AttnProcessorDistReciprocal 一致（train 默认 4）。",
    )
    parser.add_argument(
        "--segment_num",
        type=int,
        default=5,
        help="与训练 AttnProcessorDistReciprocal 一致（train 默认 5）。",
    )
    parser.add_argument(
        "--use_dataset_k",
        action="store_true",
        help="In strict file this is always enforced; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--no_save_intermediate",
        action="store_true",
        help="默认将每个去噪子步后的 latents 解码并保存到子目录；加此标志则只保存最终 jpg。",
    )
    parser.add_argument(
        "--save_initial_encode",
        action="store_true",
        help="保存 VAE 编码后、去噪前的图 step000_init_encode.*（需未加 --no_save_intermediate）。",
    )
    parser.add_argument(
        "--step_image_ext",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="中间步保存格式。",
    )
    
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    if not (0.0 <= args.pisa_ratio_start <= 1.0 and 0.0 <= args.pisa_ratio_end <= 1.0):
        raise ValueError("--pisa_ratio_start and --pisa_ratio_end must be in [0, 1].")
    if not (0.0 <= args.step_k_start <= 1.0 and 0.0 <= args.step_k_end <= 1.0):
        raise ValueError("--step_k_start and --step_k_end must be in [0, 1].")
    args.step_update_scales = parse_step_update_scales(args.step_update_scales)
    if args.step_update_scales is not None and len(args.step_update_scales) != len(args.train_T_list):
        raise ValueError(
            f"--step_update_scales expects {len(args.train_T_list)} values for train_T_list={args.train_T_list}, "
            f"but got {len(args.step_update_scales)}."
        )
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.test_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def main():
    print("main...")
    args = parse_args()
    print("args...")
    
    import os
    dist_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
    for v in dist_vars:
        if v in os.environ:
            del os.environ[v]
            print(f">>> 已清除环境变量: {v}")

    # 初始化 Accelerator
    # 显式指定 cpu=False 和 mixed_precision，不依赖自动配置
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=False
    )
    
    print(f">>> [3] Accelerator 初始化完成！设备: {accelerator.device}")

    print("accelerator = Accelerator...")
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print("logging.basicConfig...")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    print("load tokenizer...")
    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, use_safetensors=True,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,use_safetensors=True,
    ).eval()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True,
    ).eval()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    vae.enable_tiling()
    text_encoder_1.eval()
    text_encoder_2.eval()
    logging.disable(logging.CRITICAL) # Filter out warnings

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = TiledStableDiffusionXLPipeline(
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    val_dataset = TestDataset(args.test_data_dir,
        tokenizer_1=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        organization = args.organization,
        split="inference",
        upsample=args.upsample,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True
    )

    ckpt_dir = args.resume_from_checkpoint
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise ValueError(
            f"--resume_from_checkpoint must be an existing directory with LoRA+VAE weights, got: {ckpt_dir!r}"
        )
    accelerator.print(f"Loading weights from {ckpt_dir}")
    pipeline.load_lora_weights(ckpt_dir, weight_name="pytorch_lora_weights.safetensors")
    vae_path = os.path.join(ckpt_dir, "vae.ckpt")
    try:
        vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
    except TypeError:
        vae_sd = torch.load(vae_path, map_location="cpu")
    pipeline.vae.load_state_dict(vae_sd, strict=False)


    # pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet)
    # pipeline, val_dataloader = accelerator.prepare(pipeline, val_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    pipeline = pipeline.to(accelerator.device)


    fn_recursive_attn_processor(
        "unet",
        pipeline.unet,
        AttnProcessorDistReciprocal(
            hard=1e7,
            supersampling_num=args.supersampling_num,
            segment_num=args.segment_num,
            train=False,
        ),
    )
    output_dir = os.path.join(args.output_dir, args.data_id)
    os.makedirs(output_dir, exist_ok=True)

    durations = []

    for batch in tqdm(val_dataloader):
        torch.cuda.empty_cache()

        blur_strength_list = [0.2, 0.4, 0.6, 0.8, 1.0]

        for blur_strength in blur_strength_list:
            defocus_strength = batch["defocus_strength"] * blur_strength
            disparity = batch["disparity"]
            # strict：强制使用 dataset K，与 train 的 disp_coc 第二通道一致
            if "K" not in batch or batch["K"] is None:
                raise ValueError(
                    "Strict inference requires batch['K']; please use a dataset format that provides K."
                )
            k = batch["K"].to(device=accelerator.device, dtype=torch.float32)
            while k.dim() < 4:
                k = k.unsqueeze(-1)
            coc2 = defocus_strength * k
            h, w = coc2.shape[-2:]
            new_w, new_h = int(w / args.upsample), int(h / args.upsample)
            step_dir = None
            if not args.no_save_intermediate:
                step_dir = os.path.join(
                    output_dir,
                    f"{batch['filename'][0]}_blur{blur_strength:.2f}_pisa_step_frames",
                )

            # 推理（子步 pisa 与 train_pisa_per_step 一致；默认每步保存解码图到 step_dir）
            image_tensor, duration = log_validation(
                pipeline,
                pipeline.vae,
                args.seed,
                args.train_T_list,
                accelerator,
                validation_prompt=batch['texts'],
                aif_image=batch['pixel_values'],
                disp_coc=torch.cat([disparity, coc2], 1).to(weight_dtype),
                gamma=args.gamma,
                pisa_ratio_start=args.pisa_ratio_start,
                pisa_ratio_end=args.pisa_ratio_end,
                step_update_scales=args.step_update_scales,
                step_k_start=args.step_k_start,
                step_k_end=args.step_k_end,
                step_save_dir=step_dir,
                resize_to=(new_w, new_h) if step_dir is not None else None,
                original_size=tuple(int(v) for v in batch["original_size"]),
                crop_top_left=tuple(int(v) for v in batch["crop_top_left"]),
                target_size=(int(args.resolution), int(args.resolution)),
                save_initial_encode=args.save_initial_encode,
                step_image_ext=args.step_image_ext,
            )
            if duration is not None:
                durations.append(duration)

            # --- tensor -> PIL.Image（对齐 train：[-1,1] -> [0,1]）---
            image_np = _vae_output_to_uint8(image_tensor)
            image = Image.fromarray(image_np)

            # strict：最终输出不做二次 resize，直接保存解码尺寸
            out_path = f"{output_dir}/{batch['filename'][0]}_blur{blur_strength:.2f}.jpg"
            image.save(out_path, subsampling=0, quality=100)


    if durations:
        print(f"Average duration: {np.mean(durations):.4f} seconds")


if __name__ == "__main__":
    main()

```
#### inference-3t-strict运行脚本
```
CUDA_VISIBLE_DEVICES=0 python inference-3t-strict.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --resume_from_checkpoint "./output0428/run1/checkpoint-20000" \
  --test_data_dir "./test_data/input/*.jpg" \
  --organization EBB \
  --output_dir "./infer-output" \
  --data_id "3t-strict-1-20000" \
  --mixed_precision no \
  --resolution 512

```

#### train-3t：修改非线性k衰减 1628
```
# Sorryqin Qin 2026 All rights reserved.
# 变体：PISA 混合系数在「同一次前向的 timesteps_list 内」逐步变化，而非仅随 global_step 计算一次。其余与 train.py 相同。
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "vision-aided-gan-main"))

import argparse
import inspect
import logging
import math

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import shutil
from pathlib import Path
from functools import partial
from glob import glob

import numpy as np
import re
from safetensors.torch import load_file
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from custom_diffusers.attention_processor import AttnProcessor2_0 # To avoid warnings
from PISA_attn_processor import AttnProcessorDistReciprocal
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,DPMSolverMultistepScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
import cv2
# from diffusers.models.controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
# from diffusers.pipelines.controlnet_xs.pipeline_controlnet_xs_sd_xl import StableDiffusionXLControlNetXSPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import cast_training_params
from dataset import OnTheFlyDataset

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
def custom_worker_init_fn(worker_id, main_process_accelerator_device):
        """
        Ensures the worker process uses the same GPU as its parent main process.
        And sets seeds if needed.
        """
        if main_process_accelerator_device.type == 'cuda':
            # print(f"Worker {worker_id} (parent rank {os.environ.get('LOCAL_RANK', 'N/A')}): Setting CUDA device to {main_process_accelerator_device.index} (maps to global GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')})")
            torch.cuda.set_device(main_process_accelerator_device.index)

def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
    # if hasattr(module, "set_processor") and ('down_blocks.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    # if hasattr(module, "set_processor") and ('up_blocks.1.attentions.2' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
        print(">>>>>" , name )
        if not isinstance(processor, dict):
            module.set_processor(processor)
        else:
            module.set_processor(processor.pop(f"{name}.processor"))

    for sub_name, child in module.named_children():
        fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)


def build_attn_processor(model_imported, **kwargs):
    processor_cls = getattr(model_imported, "AttnProcessorDistReciprocal", AttnProcessorDistReciprocal)
    try:
        call_sig = inspect.signature(processor_cls.__call__)
        supports_pisa_strength = "pisa_strength" in call_sig.parameters
    except Exception:
        supports_pisa_strength = False

    if not supports_pisa_strength:
        logger.warning(
            "Loaded AttnProcessorDistReciprocal does not declare `pisa_strength`; "
            "falling back to local PISA_attn_processor.AttnProcessorDistReciprocal."
        )
        processor_cls = AttnProcessorDistReciprocal

    return processor_cls(**kwargs)
 
def gamma_correction(image: torch.Tensor, gamma: float = 2.2, eps=1e-7, upper_clip=None):
    if gamma == 1:
        return image
    # sign = torch.sign(image)
    # return sign*(torch.abs(image)+eps)**gamma
    base = image.min().detach()
    span = (image.max()-base).detach()
    image_norm = torch.clamp((image-base)/span,eps,upper_clip).pow(gamma) # normalize to [0,1]
    return image_norm*span + base # denormalize to [base, base+span]
    return 2*(torch.clamp(image*.5+.5,eps,upper_clip).pow(gamma)) - 1

def compute_pisa_strength_denoise_step(
    sub_step_index: int,
    num_denoise_steps: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    """在单次前向内，对 timesteps_list 第 sub_step 步使用线性插值：step0->start，末步->end。"""
    if num_denoise_steps <= 1:
        return float(end_ratio)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_ratio + (end_ratio - start_ratio) * progress)


def parse_step_update_scales(raw_scales: str):
    if raw_scales is None:
        return None
    raw_scales = raw_scales.strip()
    if raw_scales == "":
        return None
    try:
        values = [float(v.strip()) for v in raw_scales.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError("--step_update_scales must be a comma-separated float list.") from exc
    if len(values) == 0:
        return None
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError("--step_update_scales values must be in [0, 1].")
    return values


def compute_step_update_scale(
    sub_step_index: int,
    num_denoise_steps: int,
    explicit_scales,
    start_scale: float,
    end_scale: float,
) -> float:
    if explicit_scales is not None:
        return float(explicit_scales[sub_step_index])
    if num_denoise_steps <= 1:
        return float(end_scale)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_scale + (end_scale - start_scale) * progress)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

def load_my_TI(pipeline, path:str):
    if os.path.isdir(path):
        files = glob(f'{path}/text_encoder_1-*.safetensors')
        path = sorted(files, key=lambda x: int(x.split(".")[-2].split('-')[-1]))[-1]
    latest_file1 = path
    latest_file2 = path.replace('text_encoder_1', 'text_encoder_2')
    for path, encoder, tokenizer in zip([latest_file1, latest_file2],[pipeline.text_encoder, pipeline.text_encoder_2], [pipeline.tokenizer, pipeline.tokenizer_2]):
        # if path.endswith("safetensors"):
        for keys, weights in load_file(path).items():
            placeholder_tokens = keys.split()
            tokenizer.add_tokens(placeholder_tokens)
            encoder.resize_token_embeddings(len(tokenizer))
            token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            encoder.get_input_embeddings().weight.data[token_ids] = weights.to(encoder.dtype)
    return
def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "textual_inversion",
    ]

    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))



def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="rank of LoRA",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--attn_file", type=str, default="PISA_attn_processor.py", help="Path to the attention processor file.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--opt_vae", type=int, default=1, help="Whether to optimize the VAE.")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--lr_power', type=float, default=1.0, help='The power to use for the polynomial LR scheduler.')
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=3,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-07, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--ckpt_attn", action="store_true", help="Whether or not to use the attn processor in the checkpoint.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--lambda_lpips", default=.1, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--gan_step", default=1000, type=int)
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--cv_type", default="clip")
    parser.add_argument("--max_grad_norm",type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=30,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--lpips",action="store_true", help="whether to use LPIPS.")
    parser.add_argument("--edge",action="store_true", help="whether to use edge loss")
    parser.add_argument("--scale_coc", action="store_true", help="whether to scale CoC by K strength.")
    parser.add_argument("--ablation1",action="store_true", help="whether to disable MY Attn Processor.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--set_grads_to_none", action="store_true", help="Whether or not to set gradients to None.")
    parser.add_argument(
        "--pisa_ratio_start",
        type=float,
        default=1.0,
        help="本脚本：timesteps_list 第一步的 PISA 比例；与 pisa_ratio_end 在子步间线性插值（非随 global_step）。",
    )
    parser.add_argument(
        "--pisa_ratio_end",
        type=float,
        default=0.6,
        help="本脚本：timesteps_list 最后一步的 PISA 比例。",
    )
    parser.add_argument(
        "--step_update_scales",
        type=str,
        default="1.0,0.35,0.1",
        help=(
            "显式指定每个去噪子步的更新系数 k_i，逗号分隔（例如 1.0,0.8,0.6）。"
            " 若为空，则使用 step_k_start->step_k_end 线性插值。"
        ),
    )
    parser.add_argument(
        "--step_k_start",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，第一个子步的更新系数。",
    )
    parser.add_argument(
        "--step_k_end",
        type=float,
        default=0.1,
        help="当未显式指定 step_update_scales 时，最后一个子步的更新系数。",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    if not (0.0 <= args.pisa_ratio_start <= 1.0 and 0.0 <= args.pisa_ratio_end <= 1.0):
        raise ValueError("--pisa_ratio_start and --pisa_ratio_end must be in [0, 1].")
    if not (0.0 <= args.step_k_start <= 1.0 and 0.0 <= args.step_k_end <= 1.0):
        raise ValueError("--step_k_start and --step_k_end must be in [0, 1].")
    args.step_update_scales = parse_step_update_scales(args.step_update_scales)

    return args

def main():
    # 日志
    args = parse_args()
    print(f"args : {args}/n")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    versions = list(map(lambda x: int(x.split('_')[-1]) if x.startswith('logs') else -1, os.listdir(args.output_dir)))
    if len(versions):
        versions = sorted(versions)[-1]+1
    else:
        versions = 0
    logging_dir = os.path.join(args.output_dir, f'{args.logging_dir}_{versions}')
    
    # 使用Accelerate 管理多GPU训练、混合精度、梯度累积
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    if not (args.resume_from_checkpoint and args.ckpt_attn):
        shutil.copy2('./train_lora_otf.py', args.output_dir)
        # dirname = args.output_dir.strip('/').split('/')[-1]
        # shutil.copy2('./PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
        
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator( # 自动用多GPU、自动混合精度训练
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"img_logs"), exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    print("dowmload CLIP tokenizer_1")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    print("dowmload CLIP tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler.set_timesteps(1)
    print("dowmload noise_scheduler")
    print(noise_scheduler)
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_karras_sigmas=True, local_files_only=True)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,subfolder="text_encoder", revision=args.revision,
    ).eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision,
    ).eval()
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, use_safetensors=True # , variant=args.variant
    ).eval()
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True, variant=args.variant,
    ).train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer 优化器设置
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # unet = UNetControlNetXSModel.from_unet(unet, controlnet)
    weight_dtype = torch.float32
    print(accelerator.mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 定义三步 timestep
    timesteps_list = [499, 300, 100]
    # timesteps_list = [499, 350, 150]
    save_timesteps = [499, 300, 100]
    SAVE_FREQ = 50
    if args.step_update_scales is not None and len(args.step_update_scales) != len(timesteps_list):
        raise ValueError(
            f"--step_update_scales expects {len(timesteps_list)} values for timesteps_list={timesteps_list}, "
            f"but got {len(args.step_update_scales)}."
        )

    pipeline = StableDiffusionXLPipeline(
            # args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    if (not args.ablation1) and not (args.ckpt_attn and args.resume_from_checkpoint):
        print(f"Using the attn file from {args.attn_file}!")
        assert os.path.exists(args.attn_file), f"The attn file {args.attn_file} does not exist!"
        from importlib import import_module
        dirname = args.output_dir.strip('/').split('/')[-1]
        shutil.copy2(f'{args.attn_file}', f'ckpt/{dirname}_processor.py')
        model_imported = import_module(f'ckpt.{dirname}_processor')
        fn_recursive_attn_processor(
            'unet',
            pipeline.unet,
            build_attn_processor(model_imported, supersampling_num=4, segment_num=5),
        )
        print("SET TO My Processor!") 

    # Dataset and DataLoaders creation:
    train_dataset = OnTheFlyDataset(
        data_root=args.train_data_dir,
        size=args.resolution, 
        center_crop=args.center_crop,
        split="train",
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        partial_worker_init_fn = partial(custom_worker_init_fn,
                                        main_process_accelerator_device=accelerator.device)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            worker_init_fn=partial_worker_init_fn,
            shuffle=True, num_workers=args.dataloader_num_workers) # , pin_memory=True, persistent_workers=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            shuffle=True, num_workers=args.dataloader_num_workers)#, pin_memory=True, persistent_workers=True)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 冻结所有 UNet 参数，只在部分模块上注入 LoRA 层
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ], # ["to_k", "to_q", "to_v", "to_out.0"]
        # exclude_modules=".*(up_blocks).*",
    )
    for param in pipeline.unet.parameters():
        param.requires_grad_(False)
    pipeline.unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    print(len(lora_layers))
    for param in pipeline.vae.parameters():
        param.requires_grad_(False)
    if args.opt_vae:
        layers_to_opt = list(pipeline.vae.encoder.conv_in.parameters()) + \
            list(pipeline.vae.encoder.mid_block.parameters())+list(pipeline.vae.encoder.conv_out.parameters())

        for param in layers_to_opt:
            param.requires_grad_(True)

    optimizer = optimizer_class(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(pipeline.unet))
    print(count_parameters(pipeline.vae))
    if args.opt_vae:
        optimizer_vae = optimizer_class(
            layers_to_opt, #list(pipeline.vae.encoder.parameters()),
            lr=.1*args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_vae = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_vae,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    scaling_factor = pipeline.vae.config.scaling_factor
    if args.lambda_gan > 0: # add discriminator module
        import vision_aided_loss
        net_disc_TorF = vision_aided_loss.Discriminator(cv_type=args.cv_type,loss_type=args.gan_loss_type, device=accelerator.device)
        net_disc_TorF.requires_grad_(True)
        net_disc_TorF.train()

        optimizer_disc = optimizer_class(
            net_disc_TorF.parameters(),
            lr=args.learning_rate*2,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer_disc,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=1,
                power=args.lr_power,
            )
        net_disc_TorF, optimizer_disc,lr_scheduler_disc = accelerator.prepare(net_disc_TorF, optimizer_disc, lr_scheduler_disc)
    pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(
        pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet
    )
    pipeline, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(pipeline, train_dataloader, optimizer, lr_scheduler)
    # pipeline.unet.to(torch.float32)
    pipeline.unet.to(weight_dtype)
    # for param in pipeline.unet.parameters():
    #     if param.requires_grad:
    #         param.data = param.to(torch.float32)
    pipeline.vae.to(torch.float32)
    if args.mixed_precision == "fp16":
        cast_training_params([pipeline.unet], dtype=torch.float32)
    if args.lpips:
        from lpips import LPIPS
        lpips_net = LPIPS(net='vgg').to(accelerator.device)
        for param in lpips_net.parameters():
            param.requires_grad_(False)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("CTRL", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split('_')[-1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)
        else:
            path = args.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            with open(os.path.join(path, "pytorch_lora_weights.safetensors"), "rb") as f:
                state_dict = safetensors.torch.load(f.read())
            set_peft_model_state_dict(pipeline.unet, state_dict)
            # pipeline.unet.load_attn_procs(path, weight_name="pytorch_lora_weights.safetensors")
            pipeline.vae.load_state_dict(torch.load(os.path.join(path, "vae.ckpt")))
            global_step = 0 #int(path.split("-")[-1].split('.')[0])
            if args.ckpt_attn:
                print("Using the attn file from ckpt folder!")
                from importlib import import_module
                dirname = path.strip('/').split('/')[-2]
                if not os.path.exists(f'ckpt/{dirname}_processor.py'):
                    shutil.copy2(f'{path}/PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
                model_imported = import_module(f'ckpt.{dirname}_processor')
                fn_recursive_attn_processor(
                    'unet',
                    pipeline.unet,
                    build_attn_processor(model_imported, hard=global_step * args.train_batch_size),
                )
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    # snr = (alphas_cumprod / (1 - alphas_cumprod))
    texts = ['an excellent photo with a large aperture']
    encoder_output_2_list = []; encoder_hidden_states_1_list = []; encoder_hidden_states_2_list = []
    for text in texts:
        input_ids_1 = pipeline.tokenizer(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
        input_ids_2 = pipeline.tokenizer_2(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer_2.model_max_length, return_tensors="pt", ).input_ids.to(accelerator.device)
        encoder_hidden_states_1 = pipeline.text_encoder(input_ids_1, output_hidden_states=True).hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2 = pipeline.text_encoder_2(
            input_ids_2.reshape(input_ids_1.shape[0], -1), output_hidden_states=True
        )
        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2_list.append(encoder_output_2[0])
        encoder_hidden_states_1_list.append(encoder_hidden_states_1)
        encoder_hidden_states_2_list.append(encoder_hidden_states_2)
    encoder_output_2_list = torch.cat(encoder_output_2_list, dim=0)
    encoder_hidden_states_1_list = torch.cat(encoder_hidden_states_1_list, dim=0)
    encoder_hidden_states_2_list = torch.cat(encoder_hidden_states_2_list, dim=0)
    del pipeline.text_encoder, pipeline.text_encoder_2, pipeline.tokenizer, pipeline.tokenizer_2
    init_buffer = {"loss": 0}
    if args.edge:
        init_buffer['lP'] = 0
    if args.lpips:
        init_buffer['lpips'] = 0
    if args.lambda_gan > 0:
        init_buffer['loss_G'] = 0
        init_buffer['loss_D/real'] = 0
        init_buffer['loss_D/fake'] = 0
    buffer = init_buffer
    l_acc = [pipeline.unet,pipeline.vae]
    if args.lambda_gan > 0:
        l_acc += [net_disc_TorF]
    torch.cuda.empty_cache()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(*l_acc):
                disparity = batch['disparity']
                defocus_strength = batch['defocus_strength']
                optimizer.zero_grad()
                if accelerator.num_processes > 1:
                    input_latents = pipeline.vae.module.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                else:
                    input_latents = pipeline.vae.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                # input_latents = input_latents.latent_dist.sample() * scaling_factor
                input_latents = input_latents.mode() * scaling_factor
                with torch.no_grad():
                    # Get the text embedding for conditioning
                    index_for_text = torch.zeros_like(batch['K'],dtype=torch.long)
                    encoder_hidden_states_1 = encoder_hidden_states_1_list[index_for_text]
                    encoder_hidden_states_2 = encoder_hidden_states_2_list[index_for_text]
                    original_size = [
                        (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    crop_top_left = [
                        (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = torch.cat([
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(args.train_batch_size)
                        ]).to(accelerator.device, dtype=torch.float32)

                    added_cond_kwargs = {"text_embeds": encoder_output_2_list[index_for_text], "time_ids": add_time_ids}
                    encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                    aif_image, bokeh_image = batch["aif"], batch["pixel_values"]
                    
                disp_coc = torch.cat([disparity, defocus_strength * batch['K'][:,None,None,None].float()],1)
                assert disp_coc.shape[1] == 2

                x = input_latents

                # adding
                intermediate_images = {}

                num_denoise = len(timesteps_list)
                for sub_i, t in enumerate(timesteps_list):
                    pisa_strength = compute_pisa_strength_denoise_step(
                        sub_step_index=sub_i,
                        num_denoise_steps=num_denoise,
                        start_ratio=args.pisa_ratio_start,
                        end_ratio=args.pisa_ratio_end,
                    )
                    bsz = x.shape[0]
                    timesteps = torch.full(
                        (bsz,),
                        t,
                        device=x.device,
                        dtype=torch.long
                    )

                    model_pred = pipeline.unet(
                        x,
                        timesteps,
                        encoder_hidden_states.to(torch.float32),
                        added_cond_kwargs=added_cond_kwargs,
                        cross_attention_kwargs={'disp_coc': disp_coc, 'pisa_strength': pisa_strength},
                    ).sample

                    alpha_prod_t = alphas_cumprod[timesteps][:, None, None, None]
                    beta_prod_t = 1 - alpha_prod_t

                    # 单步反推 + 显式步间缩放（k_i 控制该子步更新幅度）
                    x_full = (x - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                    step_k = compute_step_update_scale(
                        sub_step_index=sub_i,
                        num_denoise_steps=num_denoise,
                        explicit_scales=args.step_update_scales,
                        start_scale=args.step_k_start,
                        end_scale=args.step_k_end,
                    )
                    x = x + step_k * (x_full - x)

                    # adding
                    if t in save_timesteps:
                        with torch.no_grad():
                            if accelerator.num_processes > 1:
                                img = pipeline.vae.module.decode(x / scaling_factor, return_dict=False)[0]
                            else:
                                img = pipeline.vae.decode(x / scaling_factor, return_dict=False)[0]

                            img = gamma_correction(img, 1/args.gamma)
                            img = torch.clamp(img, -1, 1)

                            intermediate_images[t] = img.detach()

                # 最终 latent
                pred_im = x

                
                # if accelerator.num_processes>1:
                #     pred_im = pipeline.vae.module.decode(pred_im/scaling_factor, return_dict=False)[0]
                # else:
                #     pred_im = pipeline.vae.decode(pred_im/scaling_factor, return_dict=False)[0]

                
                if accelerator.num_processes > 1:
                    pred_im = pipeline.vae.module.decode(
                        pred_im / scaling_factor, return_dict=False
                    )[0]
                else:
                    pred_im = pipeline.vae.decode(
                        pred_im / scaling_factor, return_dict=False
                    )[0]

                pred_im = gamma_correction(pred_im, 1/args.gamma)

                loss_mse = torch.mean((pred_im-bokeh_image)**2)
                loss = (loss_mse).mean() * args.lambda_l2
                pred_im = torch.clamp(pred_im, -1, 1)
                """
                Multi-scale edge loss: for more details
                """
                if args.edge:
                    nabla_bokeh_1 = bokeh_image[...,:-1] - bokeh_image[...,1:]
                    nabla_bokeh_2 = bokeh_image[...,:-1,:] - bokeh_image[...,1:,:]
                    nabla_aif_1 = aif_image[...,:-1] - aif_image[...,1:]
                    nabla_aif_2 = aif_image[...,:-1,:] - aif_image[...,1:,:]
                    loss_edge = torch.mean(torch.abs(pred_im[...,:-1]-pred_im[...,1:]-nabla_bokeh_1)*((1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                        torch.mean(torch.abs(pred_im[...,:-1,:]-pred_im[...,1:,:]-nabla_bokeh_2)*((1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    for resamp in [2,3]:
                        im_s = F.interpolate(pred_im, scale_factor=1/resamp, mode='area')
                        bokeh_s = F.interpolate(bokeh_image, scale_factor=1/resamp, mode='area')
                        aif_s = F.interpolate(aif_image, scale_factor=1/resamp, mode='area')
                        nabla_bokeh_1 = bokeh_s[...,:-1] - bokeh_s[...,1:]
                        nabla_bokeh_2 = bokeh_s[...,:-1,:] - bokeh_s[...,1:,:]
                        nabla_aif_1 = aif_s[...,:-1] - aif_s[...,1:]
                        nabla_aif_2 = aif_s[...,:-1,:] - aif_s[...,1:,:]
                        # loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)) + torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2))
                        loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                            torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    loss = loss + loss_edge

                """
                LPIPS loss
                """
                if args.lpips:
                    loss_lpips = lpips_net(pred_im, bokeh_image).mean()
                    loss = loss + loss_lpips * args.lambda_lpips
                """
                Generator loss: fool the discriminator
                """
                if args.lambda_gan > 0:
                    if global_step > args.gan_step:
                        net_disc_TorF.eval()
                        lossG = net_disc_TorF(.5+.5*pred_im, for_G=True).mean()
                        loss = loss + lossG * args.lambda_gan
                    else:
                        lossG = net_disc_TorF(.5+.5*pred_im.detach(), for_G=True).mean().detach().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.opt_vae:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    accelerator.clip_grad_norm_(lora_layers, args.max_grad_norm)
                if loss!=loss:
                    print(f"NaN!!! at {step}. Exiting...")
                    exit(0)
                if args.opt_vae:
                    optimizer_vae.step()
                    lr_scheduler_vae.step()
                optimizer.step()
                lr_scheduler.step()
                
                if args.lambda_gan > 0:
                    # if global_step > args.gan_step:
                    net_disc_TorF.train()
                    """
                    Discriminator loss: fake image vs real image
                    """
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD_real = (net_disc_TorF(.5+.5*bokeh_image, for_real=True)).mean()
                    lossD_fake = net_disc_TorF(.5+.5*(pred_im.detach().clamp_(-1,1)), for_real=False).mean()
                    loss_disc = lossD_fake + lossD_real
                    accelerator.backward(loss_disc)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc_TorF.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # images = []
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", 
                                              "lpips": f"{loss_lpips.item():.3f}" if args.lpips else 0,
                                              "PSNR": f"{-10*torch.log10(loss_mse/4).mean().item():.2f}",
                                              "pisa": f"{pisa_strength:.2f}",
                                              "k": f"{step_k:.2f}"})
                    buffer['loss']+=loss_mse.detach().mean().item()
                    if args.edge:
                        buffer['lP']+=loss_edge.detach().item()
                    if args.lpips:
                        buffer['lpips']+=loss_lpips.detach().mean().item()
                    if args.lambda_gan > 0:
                        if global_step > args.gan_step+1:
                            buffer['loss_G'] += lossG.detach().item()
                        else:
                            buffer['loss_G'] += lossG
                        # if global_step > 2001:
                        buffer['loss_D/real'] += lossD_real.detach().item()
                        buffer['loss_D/fake'] += lossD_fake.detach().item()
                        
                    # progress_bar.set_postfix(**buffer)
                    if global_step % 20 == 0:
                        for k,v in buffer.items():
                            buffer[k] = v/20
                        if args.lambda_gan > 0:
                            buffer['loss_D/sum'] = buffer['loss_D/real']+buffer['loss_D/fake']
                        # buffer['loss'] /= 20 ;buffer['lpips'] /=20
                        buffer['lr'] = lr_scheduler.get_last_lr()[0]
                        accelerator.log(buffer, step=global_step)
                        buffer = init_buffer

                    if global_step % args.validation_steps == 1:
                        for bid,(im, larim, aifim) in enumerate(zip(pred_im, bokeh_image, aif_image)):
                            Image.fromarray(np.uint8(255*(torch.cat([im,larim,aifim],-1)*.5+.5).clamp_(0,1).detach().permute(1,2,0).cpu().numpy())).save(f'{args.output_dir}/img_logs/{(global_step//args.validation_steps):04d}_{bid}_PGI.jpg')
                            
                    # adding
                    if global_step % SAVE_FREQ == 0:
                        for t, imgs in intermediate_images.items():
                            for bid, img in enumerate(imgs):
                                save_path = f"{args.output_dir}/img_logs/step{global_step}_t{t}_b{bid}.jpg"

                                Image.fromarray(
                                    np.uint8(
                                        255 * (img * 0.5 + 0.5)
                                        .clamp(0, 1)
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                ).save(save_path)


                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            # checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            checkpoints = [
                                c for c in checkpoints
                                if c.startswith("checkpoint-") and c.split("-")[-1].isdigit()
                            ]

                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))


                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        # accelerator.save_state(save_path)
                        # Save pipeline.vae
                        if accelerator.num_processes > 1:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.module.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.module.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.module.save_attn_procs(save_path)
                        else:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.save_attn_procs(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

```

#### train-5t：修改非线性衰减 1628
```
# Sorryqin Qin 2026 All rights reserved.
# 变体：PISA 混合系数在「同一次前向的 timesteps_list 内」逐步变化；本文件固定 timesteps_list=[499,400,300,200,100]（5 子步）。其余与 train_pisa_per_step.py 相同。
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "vision-aided-gan-main"))

import argparse
import inspect
import logging
import math

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import shutil
from pathlib import Path
from functools import partial
from glob import glob

import numpy as np
import re
from safetensors.torch import load_file
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from custom_diffusers.attention_processor import AttnProcessor2_0 # To avoid warnings
from PISA_attn_processor import AttnProcessorDistReciprocal
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,DPMSolverMultistepScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
import cv2
# from diffusers.models.controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
# from diffusers.pipelines.controlnet_xs.pipeline_controlnet_xs_sd_xl import StableDiffusionXLControlNetXSPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import cast_training_params
from dataset import OnTheFlyDataset

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
def custom_worker_init_fn(worker_id, main_process_accelerator_device):
        """
        Ensures the worker process uses the same GPU as its parent main process.
        And sets seeds if needed.
        """
        if main_process_accelerator_device.type == 'cuda':
            # print(f"Worker {worker_id} (parent rank {os.environ.get('LOCAL_RANK', 'N/A')}): Setting CUDA device to {main_process_accelerator_device.index} (maps to global GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')})")
            torch.cuda.set_device(main_process_accelerator_device.index)

def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
    # if hasattr(module, "set_processor") and ('down_blocks.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    # if hasattr(module, "set_processor") and ('up_blocks.1.attentions.2' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
        print(">>>>>" , name )
        if not isinstance(processor, dict):
            module.set_processor(processor)
        else:
            module.set_processor(processor.pop(f"{name}.processor"))

    for sub_name, child in module.named_children():
        fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)


def build_attn_processor(model_imported, **kwargs):
    processor_cls = getattr(model_imported, "AttnProcessorDistReciprocal", AttnProcessorDistReciprocal)
    try:
        call_sig = inspect.signature(processor_cls.__call__)
        supports_pisa_strength = "pisa_strength" in call_sig.parameters
    except Exception:
        supports_pisa_strength = False

    if not supports_pisa_strength:
        logger.warning(
            "Loaded AttnProcessorDistReciprocal does not declare `pisa_strength`; "
            "falling back to local PISA_attn_processor.AttnProcessorDistReciprocal."
        )
        processor_cls = AttnProcessorDistReciprocal

    return processor_cls(**kwargs)
 
def gamma_correction(image: torch.Tensor, gamma: float = 2.2, eps=1e-7, upper_clip=None):
    if gamma == 1:
        return image
    # sign = torch.sign(image)
    # return sign*(torch.abs(image)+eps)**gamma
    base = image.min().detach()
    span = (image.max()-base).detach()
    image_norm = torch.clamp((image-base)/span,eps,upper_clip).pow(gamma) # normalize to [0,1]
    return image_norm*span + base # denormalize to [base, base+span]
    return 2*(torch.clamp(image*.5+.5,eps,upper_clip).pow(gamma)) - 1

def compute_pisa_strength_denoise_step(
    sub_step_index: int,
    num_denoise_steps: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    """在单次前向内，对 timesteps_list 第 sub_step 步使用线性插值：step0->start，末步->end。"""
    if num_denoise_steps <= 1:
        return float(end_ratio)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_ratio + (end_ratio - start_ratio) * progress)


def parse_step_update_scales(raw_scales: str):
    if raw_scales is None:
        return None
    raw_scales = raw_scales.strip()
    if raw_scales == "":
        return None
    try:
        values = [float(v.strip()) for v in raw_scales.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise ValueError("--step_update_scales must be a comma-separated float list.") from exc
    if len(values) == 0:
        return None
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError("--step_update_scales values must be in [0, 1].")
    return values


def compute_step_update_scale(
    sub_step_index: int,
    num_denoise_steps: int,
    explicit_scales,
    start_scale: float,
    end_scale: float,
) -> float:
    if explicit_scales is not None:
        return float(explicit_scales[sub_step_index])
    if num_denoise_steps <= 1:
        return float(end_scale)
    progress = min(
        max(sub_step_index / float(num_denoise_steps - 1), 0.0),
        1.0,
    )
    return float(start_scale + (end_scale - start_scale) * progress)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

def load_my_TI(pipeline, path:str):
    if os.path.isdir(path):
        files = glob(f'{path}/text_encoder_1-*.safetensors')
        path = sorted(files, key=lambda x: int(x.split(".")[-2].split('-')[-1]))[-1]
    latest_file1 = path
    latest_file2 = path.replace('text_encoder_1', 'text_encoder_2')
    for path, encoder, tokenizer in zip([latest_file1, latest_file2],[pipeline.text_encoder, pipeline.text_encoder_2], [pipeline.tokenizer, pipeline.tokenizer_2]):
        # if path.endswith("safetensors"):
        for keys, weights in load_file(path).items():
            placeholder_tokens = keys.split()
            tokenizer.add_tokens(placeholder_tokens)
            encoder.resize_token_embeddings(len(tokenizer))
            token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            encoder.get_input_embeddings().weight.data[token_ids] = weights.to(encoder.dtype)
    return
def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "textual_inversion",
    ]

    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))



def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="train_pisa_per_step_5t：timesteps_list=[499,400,300,200,100]，其余与 train_pisa_per_step 相同。",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="rank of LoRA",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--attn_file", type=str, default="PISA_attn_processor.py", help="Path to the attention processor file.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--opt_vae", type=int, default=1, help="Whether to optimize the VAE.")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--lr_power', type=float, default=1.0, help='The power to use for the polynomial LR scheduler.')
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=3,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-07, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--ckpt_attn", action="store_true", help="Whether or not to use the attn processor in the checkpoint.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--lambda_lpips", default=.1, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--gan_step", default=1000, type=int)
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--cv_type", default="clip")
    parser.add_argument("--max_grad_norm",type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=30,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--lpips",action="store_true", help="whether to use LPIPS.")
    parser.add_argument("--edge",action="store_true", help="whether to use edge loss")
    parser.add_argument("--scale_coc", action="store_true", help="whether to scale CoC by K strength.")
    parser.add_argument("--ablation1",action="store_true", help="whether to disable MY Attn Processor.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--set_grads_to_none", action="store_true", help="Whether or not to set gradients to None.")
    parser.add_argument(
        "--pisa_ratio_start",
        type=float,
        default=1.0,
        help="本脚本：timesteps_list 第一步的 PISA 比例；与 pisa_ratio_end 在子步间线性插值（非随 global_step）。",
    )
    parser.add_argument(
        "--pisa_ratio_end",
        type=float,
        default=0.6,
        help="本脚本：timesteps_list 最后一步的 PISA 比例。",
    )
    parser.add_argument(
        "--step_update_scales",
        type=str,
        default="1.0,0.65,0.38,0.2,0.1",
        help=(
            "显式指定每个去噪子步的更新系数 k_i，逗号分隔（例如 1.0,0.9,0.8,0.7,0.6）。"
            " 若为空，则使用 step_k_start->step_k_end 线性插值。"
        ),
    )
    parser.add_argument(
        "--step_k_start",
        type=float,
        default=1.0,
        help="当未显式指定 step_update_scales 时，第一个子步的更新系数。",
    )
    parser.add_argument(
        "--step_k_end",
        type=float,
        default=0.1,
        help="当未显式指定 step_update_scales 时，最后一个子步的更新系数。",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    if not (0.0 <= args.pisa_ratio_start <= 1.0 and 0.0 <= args.pisa_ratio_end <= 1.0):
        raise ValueError("--pisa_ratio_start and --pisa_ratio_end must be in [0, 1].")
    if not (0.0 <= args.step_k_start <= 1.0 and 0.0 <= args.step_k_end <= 1.0):
        raise ValueError("--step_k_start and --step_k_end must be in [0, 1].")
    args.step_update_scales = parse_step_update_scales(args.step_update_scales)

    return args

def main():
    # 日志
    args = parse_args()
    print(f"args : {args}/n")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    versions = list(map(lambda x: int(x.split('_')[-1]) if x.startswith('logs') else -1, os.listdir(args.output_dir)))
    if len(versions):
        versions = sorted(versions)[-1]+1
    else:
        versions = 0
    logging_dir = os.path.join(args.output_dir, f'{args.logging_dir}_{versions}')
    
    # 使用Accelerate 管理多GPU训练、混合精度、梯度累积
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    if not (args.resume_from_checkpoint and args.ckpt_attn):
        shutil.copy2('./train_lora_otf.py', args.output_dir)
        # dirname = args.output_dir.strip('/').split('/')[-1]
        # shutil.copy2('./PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
        
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator( # 自动用多GPU、自动混合精度训练
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"img_logs"), exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    print("dowmload CLIP tokenizer_1")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    print("dowmload CLIP tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler.set_timesteps(1)
    print("dowmload noise_scheduler")
    print(noise_scheduler)
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_karras_sigmas=True, local_files_only=True)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,subfolder="text_encoder", revision=args.revision,
    ).eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision,
    ).eval()
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, use_safetensors=True # , variant=args.variant
    ).eval()
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True, variant=args.variant,
    ).train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer 优化器设置
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # unet = UNetControlNetXSModel.from_unet(unet, controlnet)
    weight_dtype = torch.float32
    print(accelerator.mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 定义三步 timestep
    timesteps_list = [499, 400, 300, 200, 100]
    save_timesteps = [499, 400, 300, 200, 100]
    SAVE_FREQ = 50
    if args.step_update_scales is not None and len(args.step_update_scales) != len(timesteps_list):
        raise ValueError(
            f"--step_update_scales expects {len(timesteps_list)} values for timesteps_list={timesteps_list}, "
            f"but got {len(args.step_update_scales)}."
        )

    pipeline = StableDiffusionXLPipeline(
            # args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    if (not args.ablation1) and not (args.ckpt_attn and args.resume_from_checkpoint):
        print(f"Using the attn file from {args.attn_file}!")
        assert os.path.exists(args.attn_file), f"The attn file {args.attn_file} does not exist!"
        from importlib import import_module
        dirname = args.output_dir.strip('/').split('/')[-1]
        shutil.copy2(f'{args.attn_file}', f'ckpt/{dirname}_processor.py')
        model_imported = import_module(f'ckpt.{dirname}_processor')
        fn_recursive_attn_processor(
            'unet',
            pipeline.unet,
            build_attn_processor(model_imported, supersampling_num=4, segment_num=5),
        )
        print("SET TO My Processor!") 

    # Dataset and DataLoaders creation:
    train_dataset = OnTheFlyDataset(
        data_root=args.train_data_dir,
        size=args.resolution, 
        center_crop=args.center_crop,
        split="train",
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        partial_worker_init_fn = partial(custom_worker_init_fn,
                                        main_process_accelerator_device=accelerator.device)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            worker_init_fn=partial_worker_init_fn,
            shuffle=True, num_workers=args.dataloader_num_workers) # , pin_memory=True, persistent_workers=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            shuffle=True, num_workers=args.dataloader_num_workers)#, pin_memory=True, persistent_workers=True)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 冻结所有 UNet 参数，只在部分模块上注入 LoRA 层
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ], # ["to_k", "to_q", "to_v", "to_out.0"]
        # exclude_modules=".*(up_blocks).*",
    )
    for param in pipeline.unet.parameters():
        param.requires_grad_(False)
    pipeline.unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    print(len(lora_layers))
    for param in pipeline.vae.parameters():
        param.requires_grad_(False)
    if args.opt_vae:
        layers_to_opt = list(pipeline.vae.encoder.conv_in.parameters()) + \
            list(pipeline.vae.encoder.mid_block.parameters())+list(pipeline.vae.encoder.conv_out.parameters())

        for param in layers_to_opt:
            param.requires_grad_(True)

    optimizer = optimizer_class(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(pipeline.unet))
    print(count_parameters(pipeline.vae))
    if args.opt_vae:
        optimizer_vae = optimizer_class(
            layers_to_opt, #list(pipeline.vae.encoder.parameters()),
            lr=.1*args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_vae = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_vae,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    scaling_factor = pipeline.vae.config.scaling_factor
    if args.lambda_gan > 0: # add discriminator module
        import vision_aided_loss
        net_disc_TorF = vision_aided_loss.Discriminator(cv_type=args.cv_type,loss_type=args.gan_loss_type, device=accelerator.device)
        net_disc_TorF.requires_grad_(True)
        net_disc_TorF.train()

        optimizer_disc = optimizer_class(
            net_disc_TorF.parameters(),
            lr=args.learning_rate*2,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer_disc,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=1,
                power=args.lr_power,
            )
        net_disc_TorF, optimizer_disc,lr_scheduler_disc = accelerator.prepare(net_disc_TorF, optimizer_disc, lr_scheduler_disc)
    pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(
        pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet
    )
    pipeline, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(pipeline, train_dataloader, optimizer, lr_scheduler)
    # pipeline.unet.to(torch.float32)
    pipeline.unet.to(weight_dtype)
    # for param in pipeline.unet.parameters():
    #     if param.requires_grad:
    #         param.data = param.to(torch.float32)
    pipeline.vae.to(torch.float32)
    if args.mixed_precision == "fp16":
        cast_training_params([pipeline.unet], dtype=torch.float32)
    if args.lpips:
        from lpips import LPIPS
        lpips_net = LPIPS(net='vgg').to(accelerator.device)
        for param in lpips_net.parameters():
            param.requires_grad_(False)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("CTRL", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split('_')[-1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)
        else:
            path = args.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            with open(os.path.join(path, "pytorch_lora_weights.safetensors"), "rb") as f:
                state_dict = safetensors.torch.load(f.read())
            set_peft_model_state_dict(pipeline.unet, state_dict)
            # pipeline.unet.load_attn_procs(path, weight_name="pytorch_lora_weights.safetensors")
            pipeline.vae.load_state_dict(torch.load(os.path.join(path, "vae.ckpt")))
            global_step = 0 #int(path.split("-")[-1].split('.')[0])
            if args.ckpt_attn:
                print("Using the attn file from ckpt folder!")
                from importlib import import_module
                dirname = path.strip('/').split('/')[-2]
                if not os.path.exists(f'ckpt/{dirname}_processor.py'):
                    shutil.copy2(f'{path}/PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
                model_imported = import_module(f'ckpt.{dirname}_processor')
                fn_recursive_attn_processor(
                    'unet',
                    pipeline.unet,
                    build_attn_processor(model_imported, hard=global_step * args.train_batch_size),
                )
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    # snr = (alphas_cumprod / (1 - alphas_cumprod))
    texts = ['an excellent photo with a large aperture']
    encoder_output_2_list = []; encoder_hidden_states_1_list = []; encoder_hidden_states_2_list = []
    for text in texts:
        input_ids_1 = pipeline.tokenizer(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
        input_ids_2 = pipeline.tokenizer_2(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer_2.model_max_length, return_tensors="pt", ).input_ids.to(accelerator.device)
        encoder_hidden_states_1 = pipeline.text_encoder(input_ids_1, output_hidden_states=True).hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2 = pipeline.text_encoder_2(
            input_ids_2.reshape(input_ids_1.shape[0], -1), output_hidden_states=True
        )
        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2_list.append(encoder_output_2[0])
        encoder_hidden_states_1_list.append(encoder_hidden_states_1)
        encoder_hidden_states_2_list.append(encoder_hidden_states_2)
    encoder_output_2_list = torch.cat(encoder_output_2_list, dim=0)
    encoder_hidden_states_1_list = torch.cat(encoder_hidden_states_1_list, dim=0)
    encoder_hidden_states_2_list = torch.cat(encoder_hidden_states_2_list, dim=0)
    del pipeline.text_encoder, pipeline.text_encoder_2, pipeline.tokenizer, pipeline.tokenizer_2
    init_buffer = {"loss": 0}
    if args.edge:
        init_buffer['lP'] = 0
    if args.lpips:
        init_buffer['lpips'] = 0
    if args.lambda_gan > 0:
        init_buffer['loss_G'] = 0
        init_buffer['loss_D/real'] = 0
        init_buffer['loss_D/fake'] = 0
    buffer = init_buffer
    l_acc = [pipeline.unet,pipeline.vae]
    if args.lambda_gan > 0:
        l_acc += [net_disc_TorF]
    torch.cuda.empty_cache()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(*l_acc):
                disparity = batch['disparity']
                defocus_strength = batch['defocus_strength']
                optimizer.zero_grad()
                if accelerator.num_processes > 1:
                    input_latents = pipeline.vae.module.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                else:
                    input_latents = pipeline.vae.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                # input_latents = input_latents.latent_dist.sample() * scaling_factor
                input_latents = input_latents.mode() * scaling_factor
                with torch.no_grad():
                    # Get the text embedding for conditioning
                    index_for_text = torch.zeros_like(batch['K'],dtype=torch.long)
                    encoder_hidden_states_1 = encoder_hidden_states_1_list[index_for_text]
                    encoder_hidden_states_2 = encoder_hidden_states_2_list[index_for_text]
                    original_size = [
                        (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    crop_top_left = [
                        (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = torch.cat([
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(args.train_batch_size)
                        ]).to(accelerator.device, dtype=torch.float32)

                    added_cond_kwargs = {"text_embeds": encoder_output_2_list[index_for_text], "time_ids": add_time_ids}
                    encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                    aif_image, bokeh_image = batch["aif"], batch["pixel_values"]
                    
                disp_coc = torch.cat([disparity, defocus_strength * batch['K'][:,None,None,None].float()],1)
                assert disp_coc.shape[1] == 2

                x = input_latents

                # adding
                intermediate_images = {}

                num_denoise = len(timesteps_list)
                for sub_i, t in enumerate(timesteps_list):
                    pisa_strength = compute_pisa_strength_denoise_step(
                        sub_step_index=sub_i,
                        num_denoise_steps=num_denoise,
                        start_ratio=args.pisa_ratio_start,
                        end_ratio=args.pisa_ratio_end,
                    )
                    bsz = x.shape[0]
                    timesteps = torch.full(
                        (bsz,),
                        t,
                        device=x.device,
                        dtype=torch.long
                    )

                    model_pred = pipeline.unet(
                        x,
                        timesteps,
                        encoder_hidden_states.to(torch.float32),
                        added_cond_kwargs=added_cond_kwargs,
                        cross_attention_kwargs={'disp_coc': disp_coc, 'pisa_strength': pisa_strength},
                    ).sample

                    alpha_prod_t = alphas_cumprod[timesteps][:, None, None, None]
                    beta_prod_t = 1 - alpha_prod_t

                    # 单步反推 + 显式步间缩放（k_i 控制该子步更新幅度）
                    x_full = (x - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                    step_k = compute_step_update_scale(
                        sub_step_index=sub_i,
                        num_denoise_steps=num_denoise,
                        explicit_scales=args.step_update_scales,
                        start_scale=args.step_k_start,
                        end_scale=args.step_k_end,
                    )
                    x = x + step_k * (x_full - x)

                    # adding
                    if t in save_timesteps:
                        with torch.no_grad():
                            if accelerator.num_processes > 1:
                                img = pipeline.vae.module.decode(x / scaling_factor, return_dict=False)[0]
                            else:
                                img = pipeline.vae.decode(x / scaling_factor, return_dict=False)[0]

                            img = gamma_correction(img, 1/args.gamma)
                            img = torch.clamp(img, -1, 1)

                            intermediate_images[t] = img.detach()

                # 最终 latent
                pred_im = x

                
                # if accelerator.num_processes>1:
                #     pred_im = pipeline.vae.module.decode(pred_im/scaling_factor, return_dict=False)[0]
                # else:
                #     pred_im = pipeline.vae.decode(pred_im/scaling_factor, return_dict=False)[0]

                
                if accelerator.num_processes > 1:
                    pred_im = pipeline.vae.module.decode(
                        pred_im / scaling_factor, return_dict=False
                    )[0]
                else:
                    pred_im = pipeline.vae.decode(
                        pred_im / scaling_factor, return_dict=False
                    )[0]

                pred_im = gamma_correction(pred_im, 1/args.gamma)

                loss_mse = torch.mean((pred_im-bokeh_image)**2)
                loss = (loss_mse).mean() * args.lambda_l2
                pred_im = torch.clamp(pred_im, -1, 1)
                """
                Multi-scale edge loss: for more details
                """
                if args.edge:
                    nabla_bokeh_1 = bokeh_image[...,:-1] - bokeh_image[...,1:]
                    nabla_bokeh_2 = bokeh_image[...,:-1,:] - bokeh_image[...,1:,:]
                    nabla_aif_1 = aif_image[...,:-1] - aif_image[...,1:]
                    nabla_aif_2 = aif_image[...,:-1,:] - aif_image[...,1:,:]
                    loss_edge = torch.mean(torch.abs(pred_im[...,:-1]-pred_im[...,1:]-nabla_bokeh_1)*((1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                        torch.mean(torch.abs(pred_im[...,:-1,:]-pred_im[...,1:,:]-nabla_bokeh_2)*((1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    for resamp in [2,3]:
                        im_s = F.interpolate(pred_im, scale_factor=1/resamp, mode='area')
                        bokeh_s = F.interpolate(bokeh_image, scale_factor=1/resamp, mode='area')
                        aif_s = F.interpolate(aif_image, scale_factor=1/resamp, mode='area')
                        nabla_bokeh_1 = bokeh_s[...,:-1] - bokeh_s[...,1:]
                        nabla_bokeh_2 = bokeh_s[...,:-1,:] - bokeh_s[...,1:,:]
                        nabla_aif_1 = aif_s[...,:-1] - aif_s[...,1:]
                        nabla_aif_2 = aif_s[...,:-1,:] - aif_s[...,1:,:]
                        # loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)) + torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2))
                        loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                            torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    loss = loss + loss_edge

                """
                LPIPS loss
                """
                if args.lpips:
                    loss_lpips = lpips_net(pred_im, bokeh_image).mean()
                    loss = loss + loss_lpips * args.lambda_lpips
                """
                Generator loss: fool the discriminator
                """
                if args.lambda_gan > 0:
                    if global_step > args.gan_step:
                        net_disc_TorF.eval()
                        lossG = net_disc_TorF(.5+.5*pred_im, for_G=True).mean()
                        loss = loss + lossG * args.lambda_gan
                    else:
                        lossG = net_disc_TorF(.5+.5*pred_im.detach(), for_G=True).mean().detach().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.opt_vae:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    accelerator.clip_grad_norm_(lora_layers, args.max_grad_norm)
                if loss!=loss:
                    print(f"NaN!!! at {step}. Exiting...")
                    exit(0)
                if args.opt_vae:
                    optimizer_vae.step()
                    lr_scheduler_vae.step()
                optimizer.step()
                lr_scheduler.step()
                
                if args.lambda_gan > 0:
                    # if global_step > args.gan_step:
                    net_disc_TorF.train()
                    """
                    Discriminator loss: fake image vs real image
                    """
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD_real = (net_disc_TorF(.5+.5*bokeh_image, for_real=True)).mean()
                    lossD_fake = net_disc_TorF(.5+.5*(pred_im.detach().clamp_(-1,1)), for_real=False).mean()
                    loss_disc = lossD_fake + lossD_real
                    accelerator.backward(loss_disc)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc_TorF.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # images = []
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", 
                                              "lpips": f"{loss_lpips.item():.3f}" if args.lpips else 0,
                                              "PSNR": f"{-10*torch.log10(loss_mse/4).mean().item():.2f}",
                                              "pisa": f"{pisa_strength:.2f}",
                                              "k": f"{step_k:.2f}"})
                    buffer['loss']+=loss_mse.detach().mean().item()
                    if args.edge:
                        buffer['lP']+=loss_edge.detach().item()
                    if args.lpips:
                        buffer['lpips']+=loss_lpips.detach().mean().item()
                    if args.lambda_gan > 0:
                        if global_step > args.gan_step+1:
                            buffer['loss_G'] += lossG.detach().item()
                        else:
                            buffer['loss_G'] += lossG
                        # if global_step > 2001:
                        buffer['loss_D/real'] += lossD_real.detach().item()
                        buffer['loss_D/fake'] += lossD_fake.detach().item()
                        
                    # progress_bar.set_postfix(**buffer)
                    if global_step % 20 == 0:
                        for k,v in buffer.items():
                            buffer[k] = v/20
                        if args.lambda_gan > 0:
                            buffer['loss_D/sum'] = buffer['loss_D/real']+buffer['loss_D/fake']
                        # buffer['loss'] /= 20 ;buffer['lpips'] /=20
                        buffer['lr'] = lr_scheduler.get_last_lr()[0]
                        accelerator.log(buffer, step=global_step)
                        buffer = init_buffer

                    if global_step % args.validation_steps == 1:
                        for bid,(im, larim, aifim) in enumerate(zip(pred_im, bokeh_image, aif_image)):
                            Image.fromarray(np.uint8(255*(torch.cat([im,larim,aifim],-1)*.5+.5).clamp_(0,1).detach().permute(1,2,0).cpu().numpy())).save(f'{args.output_dir}/img_logs/{(global_step//args.validation_steps):04d}_{bid}_PGI.jpg')
                            
                    # adding
                    if global_step % SAVE_FREQ == 0:
                        for t, imgs in intermediate_images.items():
                            for bid, img in enumerate(imgs):
                                save_path = f"{args.output_dir}/img_logs/step{global_step}_t{t}_b{bid}.jpg"

                                Image.fromarray(
                                    np.uint8(
                                        255 * (img * 0.5 + 0.5)
                                        .clamp(0, 1)
                                        .permute(1, 2, 0)
                                        .cpu()
                                        .numpy()
                                    )
                                ).save(save_path)


                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            # checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            checkpoints = [
                                c for c in checkpoints
                                if c.startswith("checkpoint-") and c.split("-")[-1].isdigit()
                            ]

                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))


                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        # accelerator.save_state(save_path)
                        # Save pipeline.vae
                        if accelerator.num_processes > 1:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.module.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.module.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.module.save_attn_procs(save_path)
                        else:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.save_attn_procs(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

```
#### train sh脚本 1630
```
#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./output0429/run0 ./output0429/run1 ./output0429/run2 ./output0429/run3

echo "Starting 3t training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python train-3t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --train_data_dir "./temp_data_on_the_fly/" \
  --output_dir "./output0429/run1" \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --lr_scheduler cosine_with_restarts \
  --resolution 512 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge \
  --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  > "./output0429/run1/train-3t.log" 2>&1 &

echo "Starting 3t ablation (no k decay) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python train-3t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --train_data_dir "./temp_data_on_the_fly/" \
  --output_dir "./output0429/run0" \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --lr_scheduler cosine_with_restarts \
  --resolution 512 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge \
  --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --step_update_scales "" \
  --step_k_start 1.0 \
  --step_k_end 1.0 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  > "./output0429/run0/train-3t-no-k-decay.log" 2>&1 &

echo "Starting 5t training on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python train-5t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --train_data_dir "./temp_data_on_the_fly/" \
  --output_dir "./output0429/run3" \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --lr_scheduler cosine_with_restarts \
  --resolution 512 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge \
  --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  > "./output0429/run3/train-5t.log" 2>&1 &

echo "Starting 5t ablation (no k decay) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python train-5t.py \
  --pretrained_model_name_or_path "./models/sdxl-base" \
  --train_data_dir "./temp_data_on_the_fly/" \
  --output_dir "./output0429/run2" \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision no \
  --lr_scheduler cosine_with_restarts \
  --resolution 512 \
  --lr_num_cycles 2 \
  --lr_power 0.5 \
  --lr_warmup_steps 20 \
  --gan_loss_type multilevel_sigmoid \
  --lambda_gan 1.0 \
  --gan_step 40000 \
  --edge \
  --lambda_lpips 2 \
  --learning_rate 3e-5 \
  --step_update_scales "" \
  --step_k_start 1.0 \
  --step_k_end 1.0 \
  --checkpointing_steps 10000 \
  --max_train_steps 70000 \
  > "./output0429/run2/train-5t-no-k-decay.log" 2>&1 &

echo "All tasks submitted. Check logs:"
echo "  ./output0429/run0/train-3t-no-k-decay.log"
echo "  ./output0429/run1/train-3t.log"
echo "  ./output0429/run2/train-5t-no-k-decay.log"
echo "  ./output0429/run3/train-5t.log"
```
