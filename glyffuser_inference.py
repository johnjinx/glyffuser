"""
Conditional Glyffuser inference (script version of `glyffuser inference.ipynb`).

For every prompt, generates one image per value in --steps (same seed) and saves a labeled grid,
exactly like the notebook. Multi-GPU: prompts are split across GPUs, each GPU holds its own copy
of the pipeline and works through its share in batches. Results do not depend on the number of
GPUs, because every image gets its own generator seeded with --seed.

Batching: with --batch_size > 1, shorter prompts are zero-padded to the longest one in the batch
(same as during training). The UNet pools over all text tokens, so this slightly changes those
images compared with --batch_size 1. Use 1 for exact notebook reproduction, larger for throughput.

Single prompt (same output as the notebook):
    python glyffuser_inference.py --prompts "a mental illness"

Many prompts on 4 GPUs:
    accelerate launch --multi_gpu --num_processes 4 glyffuser_inference.py \
        --prompts_file prompts.txt --batch_size 16 --precision bf16

Use a locally trained model instead of the Hugging Face one:
    python glyffuser_inference.py --model glyffuser --prompts "a quiver" "sing; folksong, ballad"
"""

import argparse
import os
import re

import torch
from accelerate import PartialState
from diffusers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from glyffuser_utils import GlyffuserPipeline, make_labeled_grid


def parse_args():
    p = argparse.ArgumentParser(description="Glyffuser inference (multi-GPU)")
    p.add_argument("--model", default="yuewu/glyffuser", help="HF repo id or local pipeline dir")
    p.add_argument("--text_encoder", default="google-t5/t5-small")
    p.add_argument("--prompts", nargs="+", default=["a mental illness"])
    p.add_argument("--prompts_file", default=None, help="Text file, one prompt per line (overrides --prompts)")
    p.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20],
                   help="num_inference_steps values; one image per value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Prompts per forward pass on each GPU (>1 is faster but pads prompts, see docstring)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="fp16/bf16 run the UNet under autocast (faster, tiny numerical differences)")
    p.add_argument("--output_dir", default="inference_outputs")
    p.add_argument("--save_individual", action="store_true", help="Also save each image separately")
    p.add_argument("--font_path", default=None, help="Font for grid labels (default PIL font)")
    p.add_argument("--font_size", type=int, default=30)
    p.add_argument("--margin", type=int, default=20)
    return p.parse_args()


def slugify(text, max_len=40):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:max_len] or "prompt"


def main():
    args = parse_args()
    state = PartialState()
    if state.device.type == "cuda":
        # t5.py moves the text encoder with `.cuda()`, i.e. to the *current* device.
        torch.cuda.set_device(state.device)

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = args.prompts
    jobs = list(enumerate(prompts))  # keep global index for stable file names

    # Download / load once on the main process first, so ranks don't race on the HF cache.
    with state.main_process_first():
        pipeline = GlyffuserPipeline.from_pretrained(args.model)
    pipeline.scheduler = DPMSolverMultistepScheduler()
    pipeline = pipeline.to(state.device)
    pipeline.set_progress_bar_config(disable=True)

    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    if autocast_dtype is not None:
        # Autocast only the UNet: T5 is prone to fp16 overflow, so text encoding stays in fp32.
        unet_forward = pipeline.unet.forward

        def autocast_forward(*f_args, **f_kwargs):
            with torch.autocast(device_type=state.device.type, dtype=autocast_dtype):
                return unet_forward(*f_args, **f_kwargs)

        pipeline.unet.forward = autocast_forward
    os.makedirs(args.output_dir, exist_ok=True)

    with state.split_between_processes(jobs) as my_jobs:
        # results[idx] = list of PIL images, one per value in args.steps
        results = {idx: [] for idx, _ in my_jobs}
        batches = [my_jobs[i:i + args.batch_size] for i in range(0, len(my_jobs), args.batch_size)]
        total = len(batches) * len(args.steps)
        bar = tqdm(total=total, desc=f"[rank {state.process_index}] sampling",
                   position=state.process_index, disable=total == 0)

        for num_steps in args.steps:
            for batch in batches:
                texts = [prompt for _, prompt in batch]
                gens = [torch.Generator(device=state.device).manual_seed(args.seed) for _ in batch]
                images = pipeline(
                    texts,
                    text_encoder=args.text_encoder,
                    generator=gens,
                    num_inference_steps=num_steps,
                ).images
                for (idx, _), img in zip(batch, images):
                    results[idx].append(img)
                bar.update(1)
        bar.close()

        for idx, prompt in my_jobs:
            stem = os.path.join(args.output_dir, f"{idx:05d}_{slugify(prompt)}")
            grid = make_labeled_grid(results[idx], prompt, args.steps,
                                     font_path=args.font_path, font_size=args.font_size, margin=args.margin)
            grid.save(stem + ".png")
            if args.save_individual:
                for n, img in zip(args.steps, results[idx]):
                    img.save(f"{stem}_steps{n}.png")

    state.wait_for_everyone()
    if state.is_main_process:
        print(f"Saved {len(prompts)} grids to {args.output_dir}")


if __name__ == "__main__":
    main()
