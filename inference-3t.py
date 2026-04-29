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
    """Convert SDXL VAE output from [-1, 1] to uint8 RGB."""
    if image.dim() == 4:
        image = image[0]
    image = (image / 2 + 0.5).clamp(0, 1)
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

    dtype = pipeline.unet.dtype  # FP16 或 FP32

    # --- 准备输入（与 train：gamma_correction 后 VAE encode）---
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

    # --- 构造 time_ids ---
    _, _, h, w = latents.shape
    height = h * 8
    width = w * 8
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        device=device,
        dtype=dtype,
    ).repeat(latents.shape[0], 1)

    # --- 其它条件输入 ---
    pooled = pooled_prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    disp_coc = disp_coc.to(device=device, dtype=dtype)

    # --- 确保 scheduler 所有 tensor 在 GPU 上 ---
    scheduler = pipeline.scheduler
    # 2026.3.5 解决NaN爆炸的问题，注释下面两行的格式转换
    scheduler.set_timesteps(1000, device=device) # adding
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    scheduler.betas = scheduler.betas.to(device=device, dtype=dtype)

    if step_save_dir is not None:
        os.makedirs(step_save_dir, exist_ok=True)
        if save_initial_encode:
            pil0 = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            _save_pil_path(
                os.path.join(step_save_dir, f"step000_init_encode.{step_image_ext}"),
                pil0,
                step_image_ext,
            )

    # --- 迭代 denoise（每步重新计算 pisa_strength，可选保存该步后的解码图）---
    n_steps = len(train_T_list)
    for sub_i, t in enumerate(train_T_list):
        pisa_strength = compute_pisa_strength_denoise_step(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            start_ratio=pisa_ratio_start,
            end_ratio=pisa_ratio_end,
        )
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)

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

        # 与 train-3t.py 保持一致：显式一步反推，不走 scheduler.step(prev_sample)
        alpha_prod_t = scheduler.alphas_cumprod[t_tensor].view(-1, 1, 1, 1).to(device=device, dtype=dtype)
        beta_prod_t = 1 - alpha_prod_t
        latents_full = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        step_k = compute_step_update_scale(
            sub_step_index=sub_i,
            num_denoise_steps=n_steps,
            explicit_scales=step_update_scales,
            start_scale=step_k_start,
            end_scale=step_k_end,
        )
        latents = latents + step_k * (latents_full - latents)

        if step_save_dir is not None:
            pil = _decode_latents_to_pil(pipeline, latents, resize_to, gamma=gamma)
            name = f"step{sub_i + 1:03d}_t{t:04d}_pisa{float(pisa_strength):.2f}.{step_image_ext}"
            _save_pil_path(os.path.join(step_save_dir, name), pil, step_image_ext)
            print(f"saved {name}")



    # --- 解码输出 ---
    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor,
        return_dict=False
    )[0]
    image = gamma_correction(image, 1.0 / gamma)

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
            # 与 train.py：disp_coc = cat(disparity, defocus_strength * K)；默认与旧 infer 一致：coc2 = defocus * (K_cli*upsample/10)
            if args.use_dataset_k and "K" in batch and batch["K"] is not None:
                k = batch["K"].to(device=accelerator.device, dtype=torch.float32)
                while k.dim() < 4:
                    k = k.unsqueeze(-1)
                coc2 = defocus_strength * k
            else:
                amplify = args.K * args.upsample / 10.0
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

            # --- tensor -> PIL.Image ---
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
