import os
import argparse
import tarfile
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from dataset import LibriSpeechWaveformDataset, GSCv2Dataset


def build_datasets(data_dir, splits):
    factory = {
        "ls100": lambda: LibriSpeechWaveformDataset(root=data_dir, url="train-clean-100", download=True),
        "ls360": lambda: LibriSpeechWaveformDataset(root=data_dir, url="train-clean-360", download=True),
        "gsc":   lambda: GSCv2Dataset(root=data_dir, download=True),
    }
    return {name: factory[name] for name in splits}


def cache_split(name, ds, out_dir, mel_transform, device, batch_size, num_workers):
    done_marker = os.path.join(out_dir, ".done")
    if os.path.isfile(done_marker):
        print(f"[{name}] complete, skipping")
        return

    os.makedirs(out_dir, exist_ok=True)
    # Per-file resume: count existing .pt files, skip exactly that many samples.
    start_idx = len([f for f in os.listdir(out_dir) if f.endswith(".pt")])
    total = len(ds)
    if start_idx >= total:
        open(done_marker, "w").close()
        print(f"[{name}] already had all {total} files, marked done")
        return
    if start_idx > 0:
        print(f"[{name}] resuming: {start_idx}/{total} already cached")

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), shuffle=False,
    )

    print(f"[{name}] caching {total} samples (from idx {start_idx})...")
    idx = 0
    for waveforms, labels in loader:
        # Skip whole batches that are entirely already-written.
        if idx + waveforms.shape[0] <= start_idx:
            idx += waveforms.shape[0]
            continue

        waveforms = waveforms.to(device)
        with torch.no_grad():
            mels = mel_transform(waveforms)
            mels = 10.0 * torch.log10(torch.clamp(mels, min=1e-10))
            mean = mels.mean(dim=[-2, -1], keepdim=True)
            std = mels.std(dim=[-2, -1], keepdim=True)
            mels = (mels - mean) / (std + 1e-6)

        for mel, label in zip(mels, labels):
            if idx >= start_idx:  # don't rewrite already-cached files
                if name == "gsc":
                    torch.save({"mel": mel.half().cpu(), "label": label.item()},
                               f"{out_dir}/{idx:07d}.pt")
                else:
                    torch.save(mel.half().cpu(), f"{out_dir}/{idx:07d}.pt")
            idx += 1

        if idx % (batch_size * 20) < batch_size:
            print(f"  [{name}] {idx}/{total}")

    open(done_marker, "w").close()
    print(f"[{name}] done. {idx} files in {out_dir}")


def main(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    cache_dir = os.path.join(args.data_dir, "mel_cache")
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    ).to(device)

    for name, make in build_datasets(args.data_dir, args.splits).items():
        ds = make()
        cache_split(name, ds, os.path.join(cache_dir, name),
                    mel_transform, device, args.batch_size, args.num_workers)

    if args.tar_to:
        os.makedirs(os.path.dirname(args.tar_to), exist_ok=True)
        print(f"Archiving cache -> {args.tar_to}")
        with tarfile.open(args.tar_to, "w") as tar:
            tar.add(cache_dir, arcname="mel_cache")
        print(f"Wrote {args.tar_to} ({os.path.getsize(args.tar_to)/1e9:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/content/data",
                   help="Local (ephemeral) dir for raw audio + mel_cache. NOT Drive.")
    p.add_argument("--splits", nargs="+", default=["ls100", "ls360", "gsc"],
                   choices=["ls100", "ls360", "gsc"],
                   help="Subset of splits. Use just 'gsc' for a fast end-to-end test.")
    p.add_argument("--tar_to", type=str, default=None,
                   help="If set, tar finished mel_cache here, e.g. "
                        "/content/drive/MyDrive/audio_ssl_data/mel_cache.tar")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()
    main(args)