import os
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_data, save_images, setup_logging


@dataclass
class SDVAETrainStats:
    loss: float
    recon_loss: float
    kl_loss: float


def decode_to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).type(torch.uint8)


def train_stable_diffusion_vae(args):
    """Fine-tune pretrained Stable Diffusion VAE on the dataset returned by get_data(args)."""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    logging.info(f"Loading pretrained VAE from {args.pretrained_vae}...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae, torch_dtype=torch.float32).to(device)
    vae.train()

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    global_step = 0
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    for epoch in range(args.epochs):
        logging.info(f"Starting SD VAE fine-tuning epoch {epoch}:")
        pbar = tqdm(dataloader)

        for images, _ in pbar:
            images = images.to(device)

            posterior = vae.encode(images)
            latents = posterior.latent_dist.sample()
            latents = latents * scaling_factor

            recon = vae.decode(latents / scaling_factor).sample

            recon_loss = F.mse_loss(recon, images)
            kl_loss = posterior.latent_dist.kl().mean()
            loss = recon_loss + args.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats = SDVAETrainStats(
                loss=loss.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_loss.item(),
            )

            pbar.set_postfix(loss=stats.loss, recon=stats.recon_loss, kl=stats.kl_loss)
            logger.add_scalar("Loss/total", stats.loss, global_step=global_step)
            logger.add_scalar("Loss/recon", stats.recon_loss, global_step=global_step)
            logger.add_scalar("Loss/kl", stats.kl_loss, global_step=global_step)
            global_step += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.add_scalar("Learning_Rate", current_lr, global_step=epoch)

        if epoch % args.save_interval == 0:
            save_reconstruction(images, recon, epoch, args.run_name)
            save_vae_checkpoint(vae, epoch, args.run_name, args.pretrained_vae)

    save_vae_checkpoint(vae, args.epochs, args.run_name, args.pretrained_vae)


def save_reconstruction(inputs: torch.Tensor, outputs: torch.Tensor, epoch: int, run_name: str):
    inputs = decode_to_uint8(inputs.detach())
    outputs = decode_to_uint8(outputs.detach())
    recon_grid = torch.cat([inputs, outputs], dim=0)
    save_images(recon_grid, os.path.join("results", run_name, f"sd_vae_recon_epoch_{epoch}.jpg"), nrow=inputs.size(0))


def save_vae_checkpoint(vae: AutoencoderKL, epoch: int, run_name: str, pretrained_id: str):
    checkpoint = {
        "state_dict": vae.state_dict(),
        "config": {
            "pretrained_id": pretrained_id,
            "scaling_factor": getattr(vae.config, "scaling_factor", 0.18215),
        },
        "epoch": epoch,
    }
    path = os.path.join("models", run_name, f"sd_vae_epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    logging.info(f"Saved SD VAE checkpoint to {path}")

