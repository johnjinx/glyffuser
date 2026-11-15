"""
Chinese glyph explorer (script version of `glyph explorer.ipynb`).

Subcommands (run in this order for the full notebook workflow):
    umap-raw     UMAP of the raw glyph pixels                       -> umap_embedding.json/.html
    train-vae    Train the VAE (multi-GPU with `accelerate launch`)  -> saved_vae/model_epoch_*.pth
    reconstruct  Original vs. reconstructed glyphs                   -> vae_reconstructions.png
    encode       VAE latent vectors + UMAP of the latent space       -> latent_vectors.npy, embedding.npy, umap_vae.*
    interp-grid  Bilinear latent interpolation between 4 characters  -> interp_grid.png
    video        Latent walk through a string of characters          -> VAE_interpolation.mp4

Examples:
    python glyph_explorer.py umap-raw --data_dir data
    accelerate launch --multi_gpu --num_processes 4 glyph_explorer.py train-vae
    python glyph_explorer.py reconstruct --vae_path saved_vae/model_epoch_200.pth
    python glyph_explorer.py encode      --vae_path saved_vae/model_epoch_200.pth
    python glyph_explorer.py interp-grid --vae_path saved_vae/model_epoch_200.pth --chars 人智工能 --steps 7
    python glyph_explorer.py video       --vae_path saved_vae/model_epoch_200.pth --chars 人工智能人

Multi-GPU: only `train-vae` benefits (DDP; `--global_batch_size` 2048 as in the notebook is split
evenly across GPUs, so results do not depend on the GPU count). UMAP runs on CPU (umap-learn), and encoding / decoding ~20k glyphs
with this small VAE takes seconds on one GPU, so those steps stay single-device.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved to files instead of shown
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_glyph_files(data_dir):
    # Sorted so that images, characters and latent vectors always line up (notebook used listdir order).
    return sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".png"))


def filename_to_character(filename):
    return chr(int(os.path.splitext(filename)[0], 16))


def character_to_filename(char):
    # The generator writes upper-case hex names (e.g. 4EBA.png). The notebook built lower-case
    # names ('4eba.png'), which only worked on case-insensitive Windows file systems.
    return f"{ord(char):04X}.png"


def _load_one(path, size, resample):
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (size, size):
            img = img.resize((size, size)) if resample is None else img.resize((size, size), resample)
        return np.asarray(img, dtype=np.uint8)


def load_glyphs(data_dir, size, resample=Image.BILINEAR, workers=16):
    """Returns (uint8 array [N, size, size], sorted file names)."""
    names = list_glyph_files(data_dir)
    paths = [os.path.join(data_dir, n) for n in names]
    with ThreadPoolExecutor(workers) as ex:
        imgs = list(tqdm(ex.map(lambda p: _load_one(p, size, resample), paths),
                         total=len(paths), desc=f"Loading {size}px glyphs"))
    return np.stack(imgs), names


def load_vae_images(data_dir, size=128):
    """VAE input images, cached as .npz next to the data (Grayscale -> Resize(128) -> [0, 255])."""
    cache = os.path.join(data_dir, f"vae_cache_{size}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], list(d["names"])
    images, names = load_glyphs(data_dir, size)
    np.savez(cache, images=images, names=np.array(names))
    return images, names


def get_device(arg):
    return torch.device(arg or ("cuda" if torch.cuda.is_available() else "cpu"))


# ---------------------------------------------------------------------------
# UMAP + plotting
# ---------------------------------------------------------------------------

def run_umap(data, n_components=2, n_neighbors=15, min_dist=0.1, seed=None):
    import umap
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(data)
    # random_state=None keeps umap-learn's parallel (multi-core) mode, as in the notebook.
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed)
    return reducer.fit_transform(scaled)


def plot_embedding(embedding, hover_text, filename, write_path):
    import plotly.express as px
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        custom_data=[hover_text],
        labels={"x": "UMAP 1", "y": "UMAP 2"},
    )
    fig.update_traces(
        marker=dict(size=3, opacity=0.2),
        hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
        hoverlabel=dict(font=dict(family="Noto Sans SC, sans-serif", size=128)),
    )
    fig.update_layout(xaxis=dict(constrain="domain"), yaxis=dict(scaleanchor="x", scaleratio=1))
    os.makedirs(write_path, exist_ok=True)
    fig.write_json(os.path.join(write_path, f"{filename}.json"))  # for embedding in the blog
    fig.write_html(os.path.join(write_path, f"{filename}.html"))  # replaces fig.show()
    print(f"Saved {filename}.json / .html to {write_path}")


# ---------------------------------------------------------------------------
# VAE (unchanged architecture)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)     # [32, 64, 64]
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)    # [64, 32, 32]
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)   # [128, 16, 16]
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # [256, 8, 8]
        self.conv5 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # [512, 4, 4]
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc_mu = nn.Linear(1024, 256)
        self.fc_logvar = nn.Linear(1024, 256)
        # Decoder
        self.fc2 = nn.Linear(256, 512 * 4 * 4)
        self.conv6 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.conv7 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv8 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv9 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv10 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = h.view(-1, 512 * 4 * 4)
        h = F.relu(self.fc1(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        h = h.view(-1, 512, 4, 4)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        return torch.sigmoid(self.conv10(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_per_sample(recon_x, x, mu, logvar):
    """BCE(sum) + KLD per sample; summing over the batch gives the notebook's loss_function."""
    bce = F.binary_cross_entropy(recon_x.float(), x.float(), reduction="none").flatten(1).sum(1)
    kld = -0.5 * torch.sum(1 + logvar.float() - mu.float().pow(2) - logvar.float().exp(), dim=1)
    return bce + kld


def load_vae(path, device):
    model = VAE().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded {path} on {device}")
    return model


def to_input(images_uint8, device):
    """uint8 [N, H, W] (numpy or tensor) -> float [N, 1, H, W] in [0, 1] (transforms.ToTensor)."""
    t = torch.as_tensor(np.asarray(images_uint8))
    return t.unsqueeze(1).to(device).float().div_(255.0)


def split_indices(n, seed, train_frac=0.8):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_umap_raw(args):
    images, names = load_glyphs(args.data_dir, 64, resample=None)  # PIL default resample, as notebook
    hover = [filename_to_character(n) for n in names]
    embedding = run_umap(images.reshape(len(images), -1).astype(np.float32), seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embedding_raw.npy"), embedding)
    plot_embedding(embedding, hover, "umap_embedding", args.plot_dir or args.out_dir)


def cmd_train_vae(args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader, Subset, TensorDataset

    # BCE is not autocast-safe, and the model is tiny: default to full precision.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    with accelerator.main_process_first():  # main builds the .npz cache, the others then read it
        images, _ = load_vae_images(args.data_dir)
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(images)))
    train_idx, test_idx = split_indices(len(dataset), args.seed)  # same 80/20 split on every rank
    per_gpu_batch = max(1, args.global_batch_size // accelerator.num_processes)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=per_gpu_batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=per_gpu_batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"{len(train_idx)} train / {len(test_idx)} test | {accelerator.num_processes} processes | "
              f"per-GPU batch {per_gpu_batch} | global batch {per_gpu_batch * accelerator.num_processes}")

    def run_epoch(loader, train):
        model.train(train)
        total = torch.zeros((), device=accelerator.device)
        with torch.set_grad_enabled(train):
            for (x,) in loader:
                x = x.unsqueeze(1).float().div_(255.0)
                recon, mu, logvar = model(x)
                per_sample = vae_loss_per_sample(recon, x, mu, logvar)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(per_sample.sum())  # notebook loss: sum over the batch
                    optimizer.step()
                # gather_for_metrics drops the samples DDP duplicated to even out the last batch
                total += accelerator.gather_for_metrics(per_sample.detach()).sum()
        return total.item()

    train_losses, test_losses = [], []
    for epoch in range(args.num_epochs):
        train_loss = run_epoch(train_loader, True) / len(train_idx)
        test_loss = run_epoch(test_loader, False) / len(test_idx)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accelerator.print(f"Epoch {epoch}, Training Loss: {train_loss}, Test Loss: {test_loss}")

        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1):
            path = os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), path)

    if accelerator.is_main_process:
        with open(os.path.join(args.save_dir, "losses.json"), "w") as f:
            json.dump({"train": train_losses, "test": test_losses}, f)
        plt.figure(figsize=(10, 5))
        epochs = range(1, args.num_epochs + 1)
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, "loss_curve.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved models, losses.json and loss_curve.png to {args.save_dir}")
    accelerator.wait_for_everyone()


@torch.no_grad()
def cmd_reconstruct(args):
    device = get_device(args.device)
    model = load_vae(args.vae_path, device)
    images, _ = load_vae_images(args.data_dir)
    _, test_idx = split_indices(len(images), args.seed)  # same held-out split as train-vae
    pick = np.random.default_rng(args.seed).choice(test_idx, size=args.num_images, replace=False)
    x = to_input(images[pick], device)
    recon, _, _ = model(x)
    x, recon = x.cpu(), recon.cpu()

    fig, axs = plt.subplots(2, args.num_images, figsize=(2.5 * args.num_images, 4))
    for i in range(args.num_images):
        axs[0, i].imshow(x[i, 0], cmap="gray")
        axs[0, i].set_title(f"Original {i + 1}")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon[i, 0], cmap="gray")
        axs[1, i].set_title(f"Reconstructed {i + 1}")
        axs[1, i].axis("off")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "vae_reconstructions.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


@torch.no_grad()
def cmd_encode(args):
    device = get_device(args.device)
    model = load_vae(args.vae_path, device)
    images, names = load_vae_images(args.data_dir)

    latents = []
    for i in tqdm(range(0, len(images), args.batch_size), desc="Encoding"):
        mu, _ = model.encode(to_input(images[i:i + args.batch_size], device))
        latents.append(mu.cpu())
    latent_vectors = torch.cat(latents).numpy()
    print(latent_vectors.shape)

    embedding = run_umap(latent_vectors, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "latent_vectors.npy"), latent_vectors)
    np.save(os.path.join(args.out_dir, "embedding.npy"), embedding)
    with open(os.path.join(args.out_dir, "latent_index.json"), "w", encoding="utf-8") as f:
        json.dump(names, f)  # row i of the .npy files <-> names[i]
    hover = [filename_to_character(n) for n in names]
    plot_embedding(embedding, hover, "umap_vae", args.plot_dir or args.out_dir)


@torch.no_grad()
def latent_vectors_for_chars(vae, chars, data_dir, device, size=128):
    vectors, found = [], []
    for c in chars:
        path = os.path.join(data_dir, character_to_filename(c))
        if not os.path.exists(path):
            print(f"Image for {c} (U+{ord(c):04X}) not found at {path}")
            continue
        img = _load_one(path, size, Image.BILINEAR)
        mu, _ = vae.encode(to_input(img[None], device))
        vectors.append(mu[0])
        found.append(c)
    return vectors, found


@torch.no_grad()
def decode_to_uint8(vae, z, batch_size=512):
    """z: [N, 256] -> uint8 numpy [N, H, W]. Decodes in batches (the notebook decoded one by one)."""
    out = []
    for i in range(0, z.shape[0], batch_size):
        out.append((vae.decode(z[i:i + batch_size])[:, 0].cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(out)


def cmd_interp_grid(args):
    device = get_device(args.device)
    vae = load_vae(args.vae_path, device)
    vecs, found = latent_vectors_for_chars(vae, args.chars, args.data_dir, device)
    if len(vecs) < 4:
        print("Could not retrieve all latent vectors. Ensure all images are available.")
        return
    tl, tr, bl, br = vecs[:4]  # chars order: top_left, top_right, bottom_left, bottom_right
    steps = args.steps
    grid = []
    for alpha in np.linspace(0, 1, steps):  # same bilinear scheme as the notebook
        top = tl * (1 - alpha) + tr * alpha
        bottom = bl * (1 - alpha) + br * alpha
        for beta in np.linspace(0, 1, steps):
            grid.append(top * (1 - beta) + bottom * beta)
    images = decode_to_uint8(vae, torch.stack(grid))

    fig, axs = plt.subplots(steps, steps, figsize=(10, 10))
    for k, img in enumerate(images):
        ax = axs[k // steps, k % steps]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "interp_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} ({''.join(found[:4])})")


def cmd_video(args):
    import cv2
    device = get_device(args.device)
    vae = load_vae(args.vae_path, device)
    vecs, found = latent_vectors_for_chars(vae, args.chars, args.data_dir, device)
    if len(vecs) < 2:
        print("Need at least two characters with images.")
        return

    alphas = torch.linspace(0, 1, args.steps_per_interpolation, device=device)[:, None]
    zs = [vecs[i] * (1 - alphas) + vecs[i + 1] * alphas for i in range(len(vecs) - 1)]
    frames = decode_to_uint8(vae, torch.cat(zs))

    h, w = frames.shape[1:]
    size = (int(w * args.scale_factor), int(h * args.scale_factor))
    os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
    video = None
    for codec in ("avc1", "X264", "mp4v"):  # H.264 if this OpenCV build has it, else MPEG-4
        video = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*codec), args.fps, size)
        if video.isOpened():
            print(f"Writing {args.video_path} with codec {codec}")
            break
        video.release()
    else:
        raise RuntimeError("No usable video codec found in this OpenCV build")

    for frame in tqdm(frames, desc="Writing video"):
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # decoded frames are single-channel
        video.write(cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR))
    video.release()
    print(f"Saved {len(frames)} frames ({''.join(found)}) to {args.video_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Chinese glyph explorer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, vae=False):
        sp.add_argument("--data_dir", default="data")
        sp.add_argument("--out_dir", default="explorer_outputs")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--device", default=None, help="Default: cuda if available")
        if vae:
            sp.add_argument("--vae_path", default="saved_vae/model_epoch_200.pth")

    sp = sub.add_parser("umap-raw", help="UMAP of raw 64x64 glyph pixels")
    common(sp)
    sp.add_argument("--plot_dir", default=None, help="Where to write the plotly JSON/HTML (default out_dir)")
    sp.set_defaults(func=cmd_umap_raw)

    sp = sub.add_parser("train-vae", help="Train the VAE (use accelerate launch for multi-GPU)")
    common(sp)
    sp.add_argument("--save_dir", default="saved_vae")
    sp.add_argument("--global_batch_size", type=int, default=2048,
                    help="Total batch over all GPUs (notebook: 2048); each GPU gets global / num_gpus")
    sp.add_argument("--learning_rate", type=float, default=1e-3)
    sp.add_argument("--num_epochs", type=int, default=200)
    sp.add_argument("--save_every", type=int, default=10)
    sp.add_argument("--num_workers", type=int, default=2)
    sp.add_argument("--mixed_precision", default="no", choices=["no", "fp16", "bf16"])
    sp.set_defaults(func=cmd_train_vae)

    sp = sub.add_parser("reconstruct", help="Original vs. reconstructed test glyphs")
    common(sp, vae=True)
    sp.add_argument("--num_images", type=int, default=6)
    sp.set_defaults(func=cmd_reconstruct)

    sp = sub.add_parser("encode", help="Latent vectors + UMAP of the latent space")
    common(sp, vae=True)
    sp.add_argument("--batch_size", type=int, default=512)
    sp.add_argument("--plot_dir", default=None, help="Where to write the plotly JSON/HTML (default out_dir)")
    sp.set_defaults(func=cmd_encode)

    sp = sub.add_parser("interp-grid", help="Bilinear interpolation between 4 characters")
    common(sp, vae=True)
    sp.add_argument("--chars", default="人智工能", help="top-left, top-right, bottom-left, bottom-right")
    sp.add_argument("--steps", type=int, default=7)
    sp.set_defaults(func=cmd_interp_grid)

    sp = sub.add_parser("video", help="Latent-space walk video through a string of characters")
    common(sp, vae=True)
    sp.add_argument("--chars", default="人工智能人")
    sp.add_argument("--video_path", default="VAE_interpolation.mp4")
    sp.add_argument("--steps_per_interpolation", type=int, default=180)
    sp.add_argument("--fps", type=int, default=60)
    sp.add_argument("--scale_factor", type=float, default=4)
    sp.set_defaults(func=cmd_video)
    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
