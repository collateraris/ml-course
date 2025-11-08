import os
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_data, setup_logging, save_images


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, eps: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.eps = eps

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: [B, C, H, W]
        b, c, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        embedding_weight = self.embedding.weight
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, embedding_weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = embedding_weight[encoding_indices].view(b, h, w, c)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()

        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        encoding_indices = encoding_indices.view(b, h, w)
        return quantized, loss, perplexity, encoding_indices

    def rebuild_codebook(self, state_dict):
        self.embedding.weight.data.copy_(state_dict["embedding.weight"])


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
        )
        self.latent_channels = latent_dim
        self.downsample_factor = 4

        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_channels, latent_dim)

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.vector_quantizer(z_e)
        return z_e, z_q, vq_loss, perplexity, indices

    def decode(self, z_q: torch.Tensor):
        return self.decoder(z_q)

    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        _, z_q, _, _, _ = self.encode(x)
        return z_q

    def decode_latents(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decode(z_q)

    def forward(self, x: torch.Tensor):
        z_e, z_q, vq_loss, perplexity, _ = self.encode(x)
        x_recon = self.decode(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return x_recon, loss, recon_loss, vq_loss, perplexity


@dataclass
class VQVAETrainStats:
    loss: float
    recon_loss: float
    vq_loss: float
    perplexity: float


def train_vqvae(args):
    """Train VQ-VAE on the dataset returned by get_data(args)."""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    model = VQVAE(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    global_step = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting VQ-VAE epoch {epoch}:")
        pbar = tqdm(dataloader)

        for images, _ in pbar:
            images = images.to(device)

            optimizer.zero_grad()
            x_recon, loss, recon_loss, vq_loss, perplexity = model(images)

            loss.backward()
            optimizer.step()

            stats = VQVAETrainStats(
                loss=loss.item(),
                recon_loss=recon_loss.item(),
                vq_loss=vq_loss.item(),
                perplexity=perplexity.item(),
            )
            pbar.set_postfix(
                loss=stats.loss,
                recon=stats.recon_loss,
                vq=stats.vq_loss,
                ppl=stats.perplexity,
            )

            logger.add_scalar("Loss/total", stats.loss, global_step=global_step)
            logger.add_scalar("Loss/recon", stats.recon_loss, global_step=global_step)
            logger.add_scalar("Loss/vq", stats.vq_loss, global_step=global_step)
            logger.add_scalar("Metrics/perplexity", stats.perplexity, global_step=global_step)
            global_step += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.add_scalar("Learning_Rate", current_lr, global_step=epoch)

        if epoch % args.save_interval == 0:
            save_recon_batch(images, x_recon, epoch, args.run_name)
            save_checkpoint(model, epoch, args.run_name)

    save_checkpoint(model, args.epochs, args.run_name)


def save_recon_batch(inputs: torch.Tensor, outputs: torch.Tensor, epoch: int, run_name: str):
    inputs = inputs.detach()
    outputs = outputs.detach()
    inputs = (inputs.clamp(-1, 1) + 1) / 2
    outputs = (outputs.clamp(-1, 1) + 1) / 2
    recon_grid = torch.cat([inputs, outputs], dim=0)
    save_images(
        (recon_grid * 255).type(torch.uint8),
        os.path.join("results", run_name, f"vqvae_recon_epoch_{epoch}.jpg"),
        nrow=inputs.size(0),
    )


def save_checkpoint(model: VQVAE, epoch: int, run_name: str):
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.config,
        "latent_channels": model.latent_channels,
        "downsample_factor": model.downsample_factor,
        "epoch": epoch,
    }
    path = os.path.join("models", run_name, f"vqvae_epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    logging.info(f"Saved VQ-VAE checkpoint to {path}")

