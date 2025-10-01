import copy
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class PatchEmbedding(nn.Module):
    """Преобразует изображения в патчи и эмбеддинги"""
    def __init__(self, img_size=64, patch_size=4, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, n_patches)
        x = x.flatten(2).transpose(1, 2)
        return x


class DiTBlock(nn.Module):
    """Блок DiT с self-attention и MLP"""
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, t_emb=None, y_emb=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


class DiT(nn.Module):
    """Diffusion Transformer для Flow Matching"""
    def __init__(self, img_size=64, patch_size=4, in_channels=3, embed_dim=512, 
                 depth=12, num_heads=8, mlp_ratio=4.0, num_classes=None, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token (для условной генерации)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Label embedding (для условной генерации)
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.label_embed = None
            
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """Создает синусоидальные эмбеддинги для времени"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
        
    def forward(self, x, t, y=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.timestep_embedding(t, self.embed_dim)
        t_emb = self.time_embed(t_emb)  # (B, embed_dim)
        
        # Label embedding
        if y is not None and self.label_embed is not None:
            y_emb = self.label_embed(y)  # (B, embed_dim)
            # Add label info to class token
            x[:, 0] = x[:, 0] + y_emb
        
        # Add time info to all tokens
        x = x + t_emb.unsqueeze(1)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, t_emb, y_emb if y is not None else None)
            
        # Final layer norm
        x = self.norm(x)
        
        # Remove class token and project to output
        x = x[:, 1:]  # Remove class token
        x = self.head(x)  # (B, num_patches, patch_size^2 * in_channels)
        
        # Reshape to image format
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x


class FlowMatchingDiT:
    def __init__(self, img_size=64, device="cuda", patch_size=4, embed_dim=512):
        """
        Flow Matching с DiT архитектурой.
        
        Args:
            img_size: размер изображений
            device: устройство для вычислений
            patch_size: размер патчей
            embed_dim: размерность эмбеддингов
        """
        self.img_size = img_size
        self.device = device
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def get_flow_path(self, x0, x1, t):
        """
        Создает путь между данными x0 и x1 в момент времени t.
        
        Args:
            x0: начальная точка (шум)
            x1: конечная точка (данные)
            t: время [0, 1]
            
        Returns:
            x_t: промежуточная точка на пути
            v_t: вектор скорости в точке x_t
        """
        # Линейная интерполяция между x0 и x1
        x_t = (1 - t) * x0 + t * x1
        
        # Вектор скорости - это направление от x0 к x1
        v_t = x1 - x0
        
        return x_t, v_t

    def sample_timesteps(self, n):
        """Сэмплирует случайные моменты времени из [0, 1]"""
        return torch.rand(n, device=self.device)

    def sample_flow_path(self, x0, x1):
        """Сэмплирует случайный путь между x0 и x1"""
        t = self.sample_timesteps(x0.shape[0])
        t = t.view(-1, 1, 1, 1)  # Добавляем размерности для broadcasting
        
        x_t, v_t = self.get_flow_path(x0, x1, t)
        return x_t, v_t, t.squeeze()

    def sample(self, model, n, labels=None, cfg_scale=0, num_steps=50):
        """
        Генерирует изображения с помощью Flow Matching + DiT.
        
        Args:
            model: обученная DiT модель
            n: количество изображений для генерации
            labels: метки классов (для условной генерации)
            cfg_scale: масштаб classifier-free guidance
            num_steps: количество шагов для численного интегрирования
        """
        logging.info(f"Sampling {n} new images using Flow Matching + DiT...")
        model.eval()
        
        with torch.no_grad():
            # Начинаем с шума
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # Численное интегрирование ODE
            dt = 1.0 / num_steps
            
            for i in tqdm(range(num_steps), position=0):
                t = torch.full((n,), i * dt, device=self.device)
                
                # Предсказываем вектор скорости
                predicted_velocity = model(x, t, labels)
                
                if cfg_scale > 0 and labels is not None:
                    # Classifier-free guidance
                    unconditional_velocity = model(x, t, None)
                    predicted_velocity = torch.lerp(unconditional_velocity, predicted_velocity, cfg_scale)
                
                # Обновляем x согласно предсказанному вектору скорости
                x = x + dt * predicted_velocity
        
        model.train()
        # Нормализуем изображения в диапазон [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def compute_loss(self, model, x0, x1, labels=None):
        """
        Вычисляет loss для Flow Matching.
        
        Args:
            model: DiT модель для предсказания вектора скорости
            x0: начальные точки (шум)
            x1: конечные точки (данные)
            labels: метки классов
        """
        # Сэмплируем случайные моменты времени и пути
        x_t, v_t, t = self.sample_flow_path(x0, x1)
        
        # Предсказываем вектор скорости
        predicted_velocity = model(x_t, t, labels)
        
        # MSE loss между истинным и предсказанным вектором скорости
        loss = nn.MSELoss()(predicted_velocity, v_t)
        
        return loss


def train_flow_matching_dit(args):
    """Функция обучения Flow Matching + DiT модели"""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Создаем DiT модель
    model = DiT(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Создаем Flow Matching объект
    flow_matching = FlowMatchingDiT(img_size=args.image_size, device=device)
    
    # Настройка логирования
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Создаем шум для начальных точек
            noise = torch.randn_like(images)
            
            # Иногда используем unconditional обучение для classifier-free guidance
            if args.conditional is False or np.random.random() < 0.1:
                labels = None
            
            # Вычисляем loss
            loss = flow_matching.compute_loss(model, noise, images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
        
        # Сохраняем результаты каждые 10 эпох
        if args.conditional is True:
            labels = torch.arange(10).long().to(device)
            sampled_images = flow_matching.sample(ema_model, n=len(labels), labels=labels, cfg_scale=3)
        else:
            sampled_images = flow_matching.sample(model, n=images.shape[0], labels=None, cfg_scale=0)
        
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch_flow_matching_dit():
    """Функция запуска обучения Flow Matching + DiT"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Основные настройки
    args.run_name = "FlowMatching_DiT"
    args.epochs = 500
    args.batch_size = 200
    args.image_size = 32
    args.dataset_path = r"E:\data_diffuse\datasets\cifar10-64\train"
    args.conditional = True
    args.num_classes = 10
    args.device = "cuda"
    args.lr = 1e-2
    args.weight_decay = 0.01
    
    # DiT настройки
    args.patch_size = 4
    args.embed_dim = 768
    args.depth = 16
    args.num_heads = 12
    args.mlp_ratio = 4.0
    args.dropout = 0.05
    
    train_flow_matching_dit(args)


if __name__ == '__main__':
    launch_flow_matching_dit()
