"""
Pipeline to fine-tune the Stable Diffusion VAE and train Flow Matching models
on the 256x256 faces dataset located at
`E:\\data_diffuse\\datasets\\faces_dataset_small\\256\\`.

Usage (defaults assume CUDA is available):
    python train_faces_flow_matching.py

You can override defaults, for example:
    python train_faces_flow_matching.py --device cpu --batch-size 2
"""

from __future__ import annotations

import argparse
import pathlib
from argparse import Namespace

from flow_matching import train_flow_matching, train_optimal_transport_flow_matching
from sd_vae_finetune import train_stable_diffusion_vae


DATASET_ROOT = pathlib.Path(r"E:\data_diffuse\datasets\faces_dataset_small\256")
PRETRAINED_VAE_ID = "stabilityai/sd-vae-ft-mse"


def build_vae_args(cfg: argparse.Namespace) -> Namespace:
    return Namespace(
        run_name=cfg.vae_run_name,
        epochs=cfg.vae_epochs,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        dataset_path=str(cfg.dataset_root),
        device=cfg.device,
        lr=cfg.vae_lr,
        weight_decay=cfg.vae_weight_decay,
        kl_weight=cfg.vae_kl_weight,
        pretrained_vae=cfg.pretrained_vae,
        save_interval=cfg.vae_save_interval,
    )


def build_flow_args(cfg: argparse.Namespace, run_name: str) -> Namespace:
    return Namespace(
        run_name=run_name,
        epochs=cfg.flow_epochs,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        dataset_path=str(cfg.dataset_root),
        conditional=False,
        num_classes=1,
        device=cfg.device,
        lr=cfg.flow_lr,
    )


def build_ot_flow_args(cfg: argparse.Namespace, vae_checkpoint: pathlib.Path) -> Namespace:
    return Namespace(
        run_name=cfg.ot_run_name,
        epochs=cfg.ot_epochs,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        dataset_path=str(cfg.dataset_root),
        conditional=False,
        num_classes=1,
        device=cfg.device,
        lr=cfg.ot_lr,
        ot_reg=cfg.ot_reg,
        ot_max_iter=cfg.ot_max_iter,
        ot_tolerance=cfg.ot_tolerance,
        vae_checkpoint=str(vae_checkpoint),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Flow Matching models on the faces dataset.")
    parser.add_argument("--device", default="cuda", help="Device to use for training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Input image resolution.")

    parser.add_argument("--vae-run-name", default="Faces256_SDVAE_Finetune", help="Run name for VAE fine-tuning.")
    parser.add_argument("--vae-epochs", type=int, default=100, help="Number of epochs for VAE fine-tuning.")
    parser.add_argument("--vae-save-interval", type=int, default=10, help="How often to save VAE checkpoints.")
    parser.add_argument("--vae-lr", type=float, default=1e-4, help="Learning rate for VAE fine-tuning.")
    parser.add_argument("--vae-weight-decay", type=float, default=0.0, help="Weight decay for VAE optimizer.")
    parser.add_argument("--vae-kl-weight", type=float, default=1e-6, help="KL loss multiplier for VAE training.")
    parser.add_argument("--pretrained-vae", default=PRETRAINED_VAE_ID, help="Pretrained Stable Diffusion VAE id.")

    parser.add_argument("--flow-run-name", default="Faces256_FlowMatching", help="Run name for pixel-space Flow Matching.")
    parser.add_argument("--flow-epochs", type=int, default=300, help="Number of epochs for pixel-space Flow Matching.")
    parser.add_argument("--flow-lr", type=float, default=1e-5, help="Learning rate for pixel-space Flow Matching.")

    parser.add_argument("--ot-run-name", default="Faces256_OTFlowMatching", help="Run name for latent Flow Matching.")
    parser.add_argument("--ot-epochs", type=int, default=500, help="Number of epochs for OT Flow Matching.")
    parser.add_argument("--ot-lr", type=float, default=5e-5, help="Learning rate for OT Flow Matching.")
    parser.add_argument("--ot-reg", type=float, default=0.1, help="Entropy regularization for Sinkhorn.")
    parser.add_argument("--ot-max-iter", type=int, default=100, help="Max iterations for Sinkhorn.")
    parser.add_argument("--ot-tolerance", type=float, default=1e-6, help="Convergence tolerance for Sinkhorn.")

    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip pixel-space Flow Matching training.",
    )
    parser.add_argument(
        "--skip-ot",
        action="store_true",
        help="Skip latent-space Flow Matching training.",
    )

    args = parser.parse_args()
    args.dataset_root = DATASET_ROOT
    return args


def main() -> None:
    cfg = parse_args()

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {cfg.dataset_root}")

    # 1. Fine-tune the Stable Diffusion VAE on the faces dataset.
    vae_args = build_vae_args(cfg)
    train_stable_diffusion_vae(vae_args)

    vae_checkpoint = pathlib.Path("models") / cfg.vae_run_name / f"sd_vae_epoch_{cfg.vae_epochs}.pt"
    if not vae_checkpoint.exists():
        raise FileNotFoundError(
            f"Expected VAE checkpoint not found at {vae_checkpoint}. "
            "Check fine-tuning configuration or adjust --vae-epochs."
        )

    # 2. Train the Flow Matching model directly in pixel space (optional).
    if not cfg.skip_flow:
        flow_args = build_flow_args(cfg, cfg.flow_run_name)
        train_flow_matching(flow_args)

    # 3. Train the Optimal Transport Flow Matching model in the VAE latent space (optional).
    if not cfg.skip_ot:
        ot_args = build_ot_flow_args(cfg, vae_checkpoint)
        train_optimal_transport_flow_matching(ot_args)


if __name__ == "__main__":
    main()

