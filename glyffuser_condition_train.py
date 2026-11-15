"""
Conditional Glyffuser training script (multi-GPU via HF Accelerate).

Converted from `glyffuser conditional.ipynb` with these main changes:
  1. T5 text embeddings and images are precomputed ONCE into a cache file,
     so the training loop never runs T5 / PIL decoding.
  2. Training runs with DDP via `accelerate launch`.
  3. Sample grids are generated in parallel: every GPU samples its share of the
     (prompt, seed) pairs and the results are gathered on the main process.

Usage (4 GPUs):
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
        glyffuser_condition_train.py --train_batch_size 16

Larger per-GPU batches need gradient checkpointing (~30% slower per sample, far less memory):
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
        glyffuser_condition_train.py --train_batch_size 64 --gradient_checkpointing

Only precompute the cache (single process, no training):
    python glyffuser_condition_train.py --precompute_only

Notes:
  * `--train_batch_size` is PER GPU. Global batch = per_gpu * num_gpus * grad_accum.
    Without gradient checkpointing a 24 GB card fits roughly 16-24 samples at 128x128.
  * `--lr_warmup_steps` is counted in optimizer steps (independent of #GPUs).
"""

import argparse
import itertools
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

import t5
from glyffuser_utils import GlyffuserPipeline, make_grid


SAMPLE_TEXTS = [
    "vicious, cruel; severely, extreme",
    "panting of cow; grunting of ox",
    "sing; folksong, ballad; rumor",
    "a quiver",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional glyffuser (multi-GPU)")
    # data
    p.add_argument("--data_dir", default="data", help="Directory containing the glyph images")
    p.add_argument("--metadata", default="data/metadata.jsonl")
    p.add_argument("--text_label", default="caption")
    p.add_argument("--image_label", default="file_name")
    p.add_argument("--cache_path", default=None,
                   help="Precomputed cache file (default: <data_dir>/cache_<encoder>_<size>.pt)")
    p.add_argument("--recompute_cache", action="store_true")
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--encode_batch_size", type=int, default=256, help="Batch size for T5 precompute")
    # model
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--text_encoder", default="google-t5/t5-small")
    p.add_argument("--cross_attention_dim", type=int, default=1024)
    # training
    p.add_argument("--train_batch_size", type=int, default=16, help="Per-GPU batch size")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--mixed_precision", default=None, choices=["no", "fp16", "bf16"],
                   help="Overrides `accelerate launch --mixed_precision` if given")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~30%% speed for much lower activation memory (allows larger batches)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--compile", action="store_true", help="torch.compile the UNet")
    p.add_argument("--seed", type=int, default=0)
    # eval / saving
    p.add_argument("--output_dir", default="glyffuser")
    p.add_argument("--save_image_epochs", type=int, default=10)
    p.add_argument("--save_model_epochs", type=int, default=30)
    p.add_argument("--sample_steps", type=int, default=50, help="DPM-Solver steps for sample grids")
    p.add_argument("--eval_images_per_prompt", type=int, default=4,
                   help="Sample grid = len(SAMPLE_TEXTS) rows x this many seeds")
    p.add_argument("--checkpoint_epochs", type=int, default=0,
                   help="Save full accelerate state every N epochs (0 = off)")
    p.add_argument("--resume_from", default=None, help="Path to an accelerate state dir to resume")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Precompute: images -> uint8 tensor, captions -> T5 embeddings (variable length)
# ---------------------------------------------------------------------------

def default_cache_path(args):
    enc = args.text_encoder.replace("/", "_")
    return os.path.join(args.data_dir, f"cache_{enc}_{args.image_size}.pt")


@torch.no_grad()
def precompute(args, cache_path, device):
    with open(args.metadata, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # 1. Images (same transform as the notebook: Resize -> 'L' -> [0, 1]); stored as uint8
    images, captions = [], []
    for rec in tqdm(records, desc="Loading images"):
        path = os.path.join(args.data_dir, rec[args.image_label])
        try:
            with Image.open(path) as img:
                img = img.convert("L").resize((args.image_size, args.image_size), Image.BILINEAR)
                images.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
        except Exception as e:  # skip broken files, like the original Collator
            print(f"Failed to process image {path}: {e}")
            continue
        captions.append(rec[args.text_label])
    images = torch.stack(images).unsqueeze(1)  # (N, 1, H, W) uint8

    # 2. Text: batch-encode with T5, then strip padding so each sample keeps its own length
    #    (identical to the original per-sample encoding, which had no padding).
    t5_model, _ = t5.get_model_and_tokenizer(args.text_encoder)
    t5_model.to(device)
    embs, lengths = [], []
    for i in tqdm(range(0, len(captions), args.encode_batch_size), desc="Encoding captions"):
        chunk = captions[i:i + args.encode_batch_size]
        enc, mask = t5.t5_encode_text(chunk, name=args.text_encoder, return_attn_mask=True)
        enc, mask = enc.float().cpu(), mask.cpu()
        for e, m in zip(enc, mask):
            n = int(m.sum())
            embs.append(e[:n])
            lengths.append(n)

    lengths = torch.tensor(lengths, dtype=torch.long)
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, 0)
    cache = {
        "images": images,
        "text_emb": torch.cat(embs).contiguous(),  # (sum(L), D) float32
        "offsets": offsets,
        "captions": captions,
        "text_encoder": args.text_encoder,
        "image_size": args.image_size,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, cache_path)
    print(f"Saved cache: {cache_path} | {len(captions)} samples | "
          f"emb dim {cache['text_emb'].shape[1]} | max len {int(lengths.max())}")

    # free T5 from GPU (it is reloaded lazily for sampling)
    t5_model.cpu()
    torch.cuda.empty_cache()


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:  # older torch without mmap/weights_only
        return torch.load(path, map_location="cpu")


class GlyphEmbeddingDataset(Dataset):
    def __init__(self, cache):
        self.images = cache["images"]
        self.text_emb = cache["text_emb"]
        self.offsets = cache["offsets"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        s, e = self.offsets[idx].item(), self.offsets[idx + 1].item()
        return self.images[idx], self.text_emb[s:e]


def collate(batch):
    images = torch.stack([b[0] for b in batch]).float().div_(255.0)  # (B, 1, H, W) in [0, 1]
    embs = [b[1] for b in batch]
    lengths = torch.tensor([e.shape[0] for e in embs])
    text = pad_sequence(embs, batch_first=True)  # pad with 0, same as the original Collator
    mask = torch.arange(text.shape[1])[None, :] < lengths[:, None]
    return images, text, mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, encoder_dim):
    return UNet2DConditionModel(
        sample_size=args.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        addition_embed_type="text",
        cross_attention_dim=args.cross_attention_dim,
        # Must equal the T5 hidden size: TextTimeEmbedding consumes the raw T5 output
        encoder_hid_dim=encoder_dim,
        encoder_hid_dim_type="text_proj",
        down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def unwrap(accelerator, model):
    # Strip DDP and torch.compile wrappers so the plain UNet can be sampled / saved.
    try:
        return accelerator.unwrap_model(model, keep_torch_compile=False)
    except TypeError:  # older accelerate
        return accelerator.unwrap_model(model)


@torch.no_grad()
def distributed_sample_grid(args, accelerator, unet, epoch):
    """Every process must call this. Jobs = (prompt, seed) pairs, split across GPUs.

    Each image has its own CPU generator, and a forward pass only ever batches copies of the
    same prompt (no padding), so the grid is identical for any number of GPUs.
    """
    n_proc, rank = accelerator.num_processes, accelerator.process_index
    k = args.eval_images_per_prompt
    jobs = [(text, args.seed + j) for text in SAMPLE_TEXTS for j in range(k)]
    per_rank = math.ceil(len(jobs) / n_proc)
    mine = jobs[rank * per_rank:(rank + 1) * per_rank]
    mine = mine + [jobs[-1]] * (per_rank - len(mine))  # pad so every rank gathers the same shape

    pipeline = GlyffuserPipeline(unet=unet, scheduler=DPMSolverMultistepScheduler())
    pipeline.set_progress_bar_config(disable=True)
    outs = []
    for text, group in itertools.groupby(mine, key=lambda job: job[0]):
        seeds = [s for _, s in group]
        outs.append(pipeline(
            [text] * len(seeds),
            text_encoder=args.text_encoder,
            generator=[torch.Generator(device="cpu").manual_seed(s) for s in seeds],
            num_inference_steps=args.sample_steps,
            output_type="np",
        ).images)  # (n, H, W, C) float in [0, 1]

    local = torch.from_numpy(np.ascontiguousarray(np.concatenate(outs))).to(accelerator.device)
    gathered = accelerator.gather(local)  # rank-major == job order
    if accelerator.is_main_process:
        arr = (gathered[:len(jobs)].cpu().numpy() * 255).round().astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        images = [Image.fromarray(a) for a in arr]
        grid = make_grid(images, rows=len(SAMPLE_TEXTS), cols=k)
        out = os.path.join(args.output_dir, "samples")
        os.makedirs(out, exist_ok=True)
        grid.save(os.path.join(out, f"{epoch:04d}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cache_path = args.cache_path or default_cache_path(args)

    if args.precompute_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precompute(args, cache_path, device)
        return

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
    if accelerator.device.type == "cuda":
        # t5.py moves T5 with `.cuda()`, i.e. to the *current* device; make that this rank's GPU.
        torch.cuda.set_device(accelerator.device)
    set_seed(args.seed, device_specific=True)  # different noise per rank

    # Precompute on the main process only; other ranks wait for the file.
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.recompute_cache or not os.path.exists(cache_path):
            precompute(args, cache_path, accelerator.device)
    accelerator.wait_for_everyone()

    # Every rank encodes prompts when sampling; load T5 now (main first) so ranks don't race on
    # the Hugging Face cache mid-training.
    with accelerator.main_process_first():
        t5.get_model_and_tokenizer(args.text_encoder)

    cache = load_cache(cache_path)
    encoder_dim = cache["text_emb"].shape[1]
    dataset = GlyphEmbeddingDataset(cache)
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

    model = build_model(args, encoder_dim)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{n_params:,} trainable parameters | {len(dataset)} samples | "
              f"{accelerator.num_processes} processes | per-GPU batch {args.train_batch_size} | "
              f"global batch {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Accelerate steps the prepared scheduler `num_processes` times per optimizer step,
    # so express both warmup and total in those units.
    steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    total_opt_steps = steps_per_epoch * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=total_opt_steps * accelerator.num_processes,
    )

    if args.compile:
        model = torch.compile(model)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers("train_conditional", config=vars(args))

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

        for clean_images, text_emb, text_mask in train_dataloader:
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_mask,  # same as inference in GlyffuserPipeline
                    return_dict=False,
                )[0]
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
            distributed_sample_grid(args, accelerator, unet, epoch)

        if accelerator.is_main_process and ((epoch + 1) % args.save_model_epochs == 0 or last):
            GlyffuserPipeline(unet=unet, scheduler=DPMSolverMultistepScheduler()).save_pretrained(
                args.output_dir)
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
