"""
Unconditional Glyffuser training + sampling (script version of `glyffuser unconditional.ipynb`).

Multi-GPU changes vs. the notebook:
  * Training runs with DDP via `accelerate launch` (was notebook_launcher, num_processes=1).
  * The image folder is packed ONCE into a uint8 cache, so data loading is not a bottleneck.
  * Sample grids (during training and with --sample_only) are generated in parallel:
    every GPU samples its share of the batch and the results are gathered on the main process.

Train (4 GPUs):
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
        glyffuser_unconditional_train.py --data_dir data --train_batch_size 32

Sample from a trained model (any number of GPUs, also works with plain `python`):
    accelerate launch --multi_gpu --num_processes 4 glyffuser_unconditional_train.py \
        --sample_only --model_path glyffuser-unconditional/epoch99 --num_samples 64

Notes:
  * `--train_batch_size` is PER GPU. Global batch = per_gpu * num_gpus * grad_accum.
  * Images are scaled to [-1, 1] as in the notebook (the conditional model uses [0, 1]).
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMPipeline, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser(description="Train / sample the unconditional glyffuser (multi-GPU)")
    # data
    p.add_argument("--data_dir", default="data128", help="Folder of glyph PNGs (searched recursively)")
    p.add_argument("--cache_path", default=None, help="Default: <data_dir>/cache_uncond_<size>.pt")
    p.add_argument("--recompute_cache", action="store_true")
    p.add_argument("--image_size", type=int, default=128)
    # training
    p.add_argument("--train_batch_size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=500, help="In optimizer steps")
    p.add_argument("--mixed_precision", default=None, choices=["no", "fp16", "bf16"],
                   help="Overrides `accelerate launch --mixed_precision` if given")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~30%% speed for much lower activation memory (allows larger batches)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--compile", action="store_true", help="torch.compile the UNet")
    p.add_argument("--seed", type=int, default=0)
    # eval / saving
    p.add_argument("--output_dir", default="glyffuser-unconditional")
    p.add_argument("--save_image_epochs", type=int, default=1)
    p.add_argument("--save_model_epochs", type=int, default=30)
    p.add_argument("--eval_batch_size", type=int, default=16, help="Total images per sample grid")
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--checkpoint_epochs", type=int, default=0,
                   help="Save full accelerate state every N epochs (0 = off)")
    p.add_argument("--resume_from", default=None, help="Path to an accelerate state dir to resume")
    # sampling only
    p.add_argument("--sample_only", action="store_true", help="Skip training, sample from --model_path")
    p.add_argument("--model_path", default=None, help="Saved DDPMPipeline dir (default: --output_dir)")
    p.add_argument("--num_samples", type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: pack the PNG folder into one uint8 tensor
# ---------------------------------------------------------------------------

def build_cache(args, cache_path):
    paths = sorted(Path(args.data_dir).glob("**/*.png"))
    images = []
    for p in tqdm(paths, desc="Loading images"):
        try:
            with Image.open(p) as img:
                img = img.convert("L")
                if img.size != (args.image_size, args.image_size):
                    img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
                images.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
        except Exception as e:
            print(f"Failed to process image {p}: {e}")
    images = torch.stack(images).unsqueeze(1)  # (N, 1, H, W) uint8
    tmp = cache_path + ".tmp"
    torch.save({"images": images, "image_size": args.image_size}, tmp)
    os.replace(tmp, cache_path)
    print(f"Saved cache: {cache_path} | {images.shape[0]} images")


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GlyphImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]


def collate(batch):
    # uint8 [0, 255] -> float [-1, 1]  (ToTensor + normalize_neg_one_to_one in the notebook)
    return torch.stack(batch).float().div_(127.5).sub_(1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args):
    return UNet2DModel(
        sample_size=args.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )


# ---------------------------------------------------------------------------
# Distributed sampling: each rank generates its share, main process assembles the grid
# ---------------------------------------------------------------------------

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


@torch.no_grad()
def distributed_sample(accelerator, unet, num_images, num_steps, seed):
    """Every process must call this. Returns a list of PIL images on the main process, else None."""
    n_proc, rank = accelerator.num_processes, accelerator.process_index
    per_rank = math.ceil(num_images / n_proc)

    pipeline = DDPMPipeline(unet=unet, scheduler=DPMSolverMultistepScheduler())
    pipeline.set_progress_bar_config(disable=not accelerator.is_main_process)
    # One CPU generator per image, seeded by its global index -> the grid is identical
    # no matter how many GPUs are used.
    gens = [torch.Generator(device="cpu").manual_seed(seed + rank * per_rank + i) for i in range(per_rank)]
    images = pipeline(
        batch_size=per_rank, generator=gens, num_inference_steps=num_steps, output_type="np",
    ).images  # (per_rank, H, W, C) float in [0, 1]

    local = torch.from_numpy(np.ascontiguousarray(images)).to(accelerator.device)
    gathered = accelerator.gather(local)  # (per_rank * n_proc, H, W, C), rank-major order
    if not accelerator.is_main_process:
        return None
    arr = (gathered[:num_images].cpu().numpy() * 255).round().astype("uint8")
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return [Image.fromarray(a) for a in arr]


def unwrap(accelerator, model):
    # Strip DDP and torch.compile wrappers so the plain UNet can be sampled / saved.
    try:
        return accelerator.unwrap_model(model, keep_torch_compile=False)
    except TypeError:  # older accelerate
        return accelerator.unwrap_model(model)


def save_grid(images, path):
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_grid(images, rows=rows, cols=cols).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sample_only(args, accelerator):
    model_path = args.model_path or args.output_dir
    unet = UNet2DModel.from_pretrained(model_path, subfolder="unet").to(accelerator.device)
    unet.eval()
    images = distributed_sample(accelerator, unet, args.num_samples, args.sample_steps, args.seed)
    if accelerator.is_main_process:
        out = os.path.join(args.output_dir, "samples", "samples.png")
        save_grid(images, out)
        print(f"Saved {len(images)} samples to {out}")


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accel_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    if args.mixed_precision is not None:
        accel_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = Accelerator(**accel_kwargs)

    if args.sample_only:
        run_sample_only(args, accelerator)
        return

    set_seed(args.seed, device_specific=True)

    cache_path = args.cache_path or os.path.join(args.data_dir, f"cache_uncond_{args.image_size}.pt")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.recompute_cache or not os.path.exists(cache_path):
            build_cache(args, cache_path)
    accelerator.wait_for_everyone()

    dataset = GlyphImageDataset(load_cache(cache_path)["images"])
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = build_model(args)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{n_params:,} trainable parameters | {len(dataset)} images | "
              f"{accelerator.num_processes} processes | per-GPU batch {args.train_batch_size} | "
              f"global batch {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # The prepared scheduler is stepped `num_processes` times per optimizer step.
    steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=steps_per_epoch * args.num_epochs * accelerator.num_processes,
    )

    if args.compile:
        model = torch.compile(model)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers("train_unconditional", config=vars(args))

    start_epoch, global_step = 0, 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        name = os.path.basename(os.path.normpath(args.resume_from))
        if name.startswith("epoch_"):
            start_epoch = int(name.split("_")[1]) + 1
            global_step = start_epoch * len(train_dataloader)
        accelerator.print(f"Resumed from {args.resume_from}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for clean_images in train_dataloader:
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()

        last = epoch == args.num_epochs - 1
        if args.checkpoint_epochs and ((epoch + 1) % args.checkpoint_epochs == 0 or last):
            accelerator.save_state(os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch}"))

        unet = unwrap(accelerator, model)
        if (epoch + 1) % args.save_image_epochs == 0 or last:
            # All ranks sample in parallel (not only the main process).
            unet.eval()
            images = distributed_sample(accelerator, unet, args.eval_batch_size, args.sample_steps, args.seed)
            if accelerator.is_main_process:
                save_grid(images, os.path.join(args.output_dir, "samples", f"{epoch:04d}.png"))

        if accelerator.is_main_process and ((epoch + 1) % args.save_model_epochs == 0 or last):
            DDPMPipeline(unet=unet, scheduler=DPMSolverMultistepScheduler()).save_pretrained(
                os.path.join(args.output_dir, f"epoch{epoch}"))
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
