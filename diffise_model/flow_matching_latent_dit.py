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
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class ImprovedVAE(nn.Module):
    """Улучшенный VAE для Flow Matching в латентном пространстве 32x32"""
    def __init__(self, in_channels=3, latent_channels=16, base_channels=64):
        super().__init__()
        
        # Encoder с residual connections
        self.encoder = nn.ModuleList([
            # 64x64 -> 32x32
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2),
            ),
            
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(base_channels, base_channels),
            
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(base_channels, base_channels),
            
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(base_channels, base_channels),
        ])
        
        # Финальный слой для изменения количества каналов
        self.encoder_final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
        )
        
        # VAE heads с attention
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.mu_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, latent_channels, 3, 1, 1),
            nn.BatchNorm2d(latent_channels),
        )
        
        self.logvar_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, latent_channels, 3, 1, 1),
            nn.BatchNorm2d(latent_channels),
        )
        
        # Decoder с residual connections
        self.decoder = nn.ModuleList([
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(latent_channels, latent_channels),
            
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(latent_channels, latent_channels),
            
            # 32x32 -> 32x32 (residual block, same channels)
            self._make_residual_block(latent_channels, latent_channels),
        ])
        
        # Финальный слой для upsampling
        self.decoder_final = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, in_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def _make_residual_block(self, in_channels, out_channels):
        """Создает residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
    
    def _apply_residual_block(self, x, block):
        """Применяет residual block"""
        residual = block(x)
        
        # Проверяем, нужно ли адаптировать размеры для residual connection
        if residual.shape == x.shape:
            # Если размеры совпадают, добавляем residual connection
            residual = residual + x
        else:
            # Если размеры не совпадают, используем только residual без skip connection
            pass
            
        return F.leaky_relu(residual, 0.2)
        
    def encode(self, x):
        """Кодирует изображение в параметры распределения (mu, logvar)"""
        # Применяем encoder с residual connections
        h = x
        
        for i, layer in enumerate(self.encoder):
            if i == 0:
                h = layer(h)
            else:
                h = self._apply_residual_block(h, layer)
        
        # Финальный слой для изменения количества каналов
        h = self.encoder_final(h)
        
        # Применяем attention
        attention_weights = self.attention(h)
        h = h * attention_weights
        
        # Получаем mu и logvar
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick для VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Декодирует латентное представление в изображение"""
        h = z
        
        # Применяем residual blocks
        for layer in self.decoder:
            h = self._apply_residual_block(h, layer)
        
        # Финальный upsampling
        h = self.decoder_final(h)
        
        return h
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar
    
    def sample(self, n_samples, device):
        """Генерирует случайные образцы из латентного пространства"""
        z = torch.randn(n_samples, self.mu_head[0].out_channels, 32, 32).to(device)
        return self.decode(z)
    
    def compute_loss(self, recon_x, x, mu, logvar, beta=1.0, perceptual_weight=0.1):
        """Вычисляет улучшенный VAE loss"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # Perceptual loss (L1 для лучшего качества)
        perceptual_loss = F.l1_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + perceptual_weight * perceptual_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class LatentPatchEmbedding(nn.Module):
    """Преобразует латентные представления 32x32 в патчи и эмбеддинги"""
    def __init__(self, latent_size=32, latent_channels=8, patch_size=4, embed_dim=512):
        super().__init__()
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.n_patches = (latent_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, latent_channels, latent_size, latent_size) -> (B, latent_channels, 32, 32)
        x = self.proj(x)  # (B, embed_dim, latent_size//patch_size, latent_size//patch_size) -> (B, embed_dim, 8, 8)
        # (B, embed_dim, 8, 8) -> (B, embed_dim, 64)
        x = x.flatten(2).transpose(1, 2)
        return x


class LatentDiTBlock(nn.Module):
    """Блок DiT для латентного пространства"""
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


class LatentDiT(nn.Module):
    """DiT для работы в латентном пространстве 32x32"""
    def __init__(self, latent_size=32, latent_channels=8, patch_size=4, embed_dim=512, 
                 depth=12, num_heads=8, mlp_ratio=4.0, num_classes=None, dropout=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (latent_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = LatentPatchEmbedding(latent_size, latent_channels, patch_size, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Label embedding
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.label_embed = None
            
        # DiT blocks
        self.blocks = nn.ModuleList([
            LatentDiTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.head = nn.Linear(embed_dim, patch_size * patch_size * latent_channels)
        
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
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            
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
        x = self.head(x)  # (B, num_patches, patch_size^2 * latent_channels)
        
        # Reshape to latent format
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, self.latent_channels)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.latent_channels, self.latent_size, self.latent_size)
        
        return x


class LatentFlowMatchingDiT:
    def __init__(self, img_size=64, latent_size=32, latent_channels=8, device="cuda"):
        """
        Flow Matching в латентном пространстве 32x32 с DiT.
        
        Args:
            img_size: размер изображений
            latent_size: размер латентного пространства (32x32)
            latent_channels: количество каналов в латентном пространстве
            device: устройство для вычислений
        """
        self.img_size = img_size
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.device = device

    def get_flow_path(self, z0, z1, t):
        """
        Создает путь между латентными представлениями z0 и z1 в момент времени t.
        
        Args:
            z0: начальная точка (шум в латентном пространстве)
            z1: конечная точка (закодированные данные)
            t: время [0, 1]
            
        Returns:
            z_t: промежуточная точка на пути
            v_t: вектор скорости в точке z_t
        """
        # Линейная интерполяция между z0 и z1
        z_t = (1 - t) * z0 + t * z1
        
        # Вектор скорости - это направление от z0 к z1
        v_t = z1 - z0
        
        return z_t, v_t

    def sample_timesteps(self, n):
        """Сэмплирует случайные моменты времени из [0, 1]"""
        return torch.rand(n, device=self.device)

    def sample_flow_path(self, z0, z1):
        """Сэмплирует случайный путь между z0 и z1"""
        t = self.sample_timesteps(z0.shape[0])
        t = t.view(-1, 1, 1, 1)  # Добавляем размерности для broadcasting
        
        z_t, v_t = self.get_flow_path(z0, z1, t)
        return z_t, v_t, t.squeeze()

    def sample(self, model, autoencoder, n, labels=None, cfg_scale=0, num_steps=50):
        """
        Генерирует изображения с помощью Flow Matching в латентном пространстве + DiT.
        
        Args:
            model: обученная DiT модель для латентного пространства
            autoencoder: обученный автоэнкодер
            n: количество изображений для генерации
            labels: метки классов (для условной генерации)
            cfg_scale: масштаб classifier-free guidance
            num_steps: количество шагов для численного интегрирования
        """
        logging.info(f"Sampling {n} new images using Latent Flow Matching + DiT...")
        model.eval()
        autoencoder.eval()
        
        with torch.no_grad():
            # Начинаем с шума в латентном пространстве
            z = torch.randn((n, self.latent_channels, self.latent_size, self.latent_size)).to(self.device)
            
            # Численное интегрирование ODE
            dt = 1.0 / num_steps
            
            for i in tqdm(range(num_steps), position=0):
                t = torch.full((n,), i * dt, device=self.device)
                
                # Предсказываем вектор скорости в латентном пространстве
                predicted_velocity = model(z, t, labels)
                
                if cfg_scale > 0 and labels is not None:
                    # Classifier-free guidance
                    unconditional_velocity = model(z, t, None)
                    predicted_velocity = torch.lerp(unconditional_velocity, predicted_velocity, cfg_scale)
                
                # Обновляем z согласно предсказанному вектору скорости
                z = z + dt * predicted_velocity
            
            # Декодируем латентное представление в изображения
            images = autoencoder.decode(z)
        
        model.train()
        autoencoder.train()
        
        # Нормализуем изображения в диапазон [0, 1]
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        return images

    def compute_loss(self, model, z0, z1, labels=None):
        """
        Вычисляет улучшенный loss для Flow Matching в латентном пространстве.
        
        Args:
            model: DiT модель для предсказания вектора скорости
            z0: начальные точки (шум в латентном пространстве)
            z1: конечные точки (закодированные данные)
            labels: метки классов
        """
        # Сэмплируем случайные моменты времени и пути
        z_t, v_t, t = self.sample_flow_path(z0, z1)
        
        # Предсказываем вектор скорости
        predicted_velocity = model(z_t, t, labels)
        
        # Улучшенный loss с несколькими компонентами
        # 1. Основной MSE loss
        mse_loss = nn.MSELoss()(predicted_velocity, v_t)
        
        # 2. L1 loss для лучшей детализации
        l1_loss = nn.L1Loss()(predicted_velocity, v_t)
        
        # 3. Cosine similarity loss для направления
        v_t_flat = v_t.view(v_t.size(0), -1)
        pred_flat = predicted_velocity.view(predicted_velocity.size(0), -1)
        cosine_loss = 1 - F.cosine_similarity(v_t_flat, pred_flat, dim=1).mean()
        
        # Комбинированный loss с более агрессивными весами
        total_loss = mse_loss + 0.5 * l1_loss + 0.3 * cosine_loss
        grad_penalty = torch.tensor(0.0, device=z_t.device)
        
        # Дополнительная диагностика
        with torch.no_grad():
            # Вычисляем статистики для диагностики
            v_t_norm = torch.norm(v_t.view(v_t.size(0), -1), dim=1).mean()
            pred_norm = torch.norm(predicted_velocity.view(predicted_velocity.size(0), -1), dim=1).mean()
            z0_norm = torch.norm(z0.view(z0.size(0), -1), dim=1).mean()
            z1_norm = torch.norm(z1.view(z1.size(0), -1), dim=1).mean()
            
            # Логируем диагностическую информацию
            if hasattr(self, '_step'):
                self._step += 1
            else:
                self._step = 0
            

        
        return total_loss
    
    def _compute_gradient_penalty(self, pred, z_t):
        """Вычисляет gradient penalty для стабильности"""
        try:
            grad_outputs = torch.ones_like(pred)
            gradients = torch.autograd.grad(
                outputs=pred,
                inputs=z_t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty
        except RuntimeError:
            # Если не удается вычислить градиенты, возвращаем 0
            return torch.tensor(0.0, device=pred.device)
    
    def evaluate_model(self, model, autoencoder, dataloader, device, num_samples=10):
        """Оценка качества модели"""
        model.eval()
        autoencoder.eval()
        
        with torch.no_grad():
            # Тестируем реконструкцию VAE
            total_recon_loss = 0
            total_kl_loss = 0
            num_batches = 0
            
            for images, _ in dataloader:
                if num_batches >= 5:  # Ограничиваем количество батчей для тестирования
                    break
                    
                images = images.to(device)
                recon_images, z, mu, logvar = autoencoder(images)
                
                # Вычисляем loss VAE
                total_loss, recon_loss, kl_loss = autoencoder.compute_loss(recon_images, images, mu, logvar)
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
            
            avg_recon_loss = total_recon_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches
            
            print(f"VAE Reconstruction Loss: {avg_recon_loss:.4f}")
            print(f"VAE KL Loss: {avg_kl_loss:.4f}")
            
            # Тестируем Flow Matching
            z0 = torch.randn(num_samples, self.latent_channels, self.latent_size, self.latent_size).to(device)
            z1 = torch.randn_like(z0)
            
            # Тестируем несколько временных шагов
            total_fm_loss = 0
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                t = torch.full((num_samples,), t_val, device=device)
                t_expanded = t.view(-1, 1, 1, 1)  # Правильное расширение размерности
                z_t, v_t = self.get_flow_path(z0, z1, t_expanded)
                
                predicted_velocity = model(z_t, t, None)
                fm_loss = nn.MSELoss()(predicted_velocity, v_t)
                total_fm_loss += fm_loss.item()
            
            avg_fm_loss = total_fm_loss / 5
            print(f"Flow Matching Average Loss: {avg_fm_loss:.4f}")
            
        model.train()
        autoencoder.train()
        
        return avg_recon_loss, avg_kl_loss, avg_fm_loss


def train_latent_flow_matching_dit(args):
    """Функция обучения Latent Flow Matching + DiT модели"""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Создаем улучшенный VAE
    autoencoder = ImprovedVAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels
    ).to(device)
    
    # Создаем DiT модель для латентного пространства
    model = LatentDiT(
        latent_size=args.latent_size,
        latent_channels=args.latent_channels,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    # Оптимизаторы с улучшенными настройками
    autoencoder_optimizer = optim.AdamW(autoencoder.parameters(), lr=args.autoencoder_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    model_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # Learning rate schedulers с warmup
    autoencoder_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(autoencoder_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    model_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Создаем Flow Matching объект
    flow_matching = LatentFlowMatchingDiT(
        img_size=args.image_size,
        latent_size=args.latent_size,
        latent_channels=args.latent_channels,
        device=device
    )
    
    # Настройка логирования
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    # Проверяем, есть ли уже обученная VAE
    vae_checkpoint_path = os.path.join("models", args.run_name, "vae_ckpt.pt")
    
    if os.path.exists(vae_checkpoint_path) and not args.retrain_vae:
        logging.info("Loading pre-trained VAE...")
        autoencoder.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
        logging.info("VAE loaded successfully!")
    else:
        # Обучаем VAE с нуля
        logging.info("Training VAE from scratch...")
        for epoch in range(args.autoencoder_epochs):
            autoencoder.train()
            pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch}")
            
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                
                # Forward pass
                recon_images, latent, mu, logvar = autoencoder(images)
                
                # VAE loss (reconstruction + perceptual + KL divergence)
                total_loss, recon_loss, kl_loss = autoencoder.compute_loss(recon_images, images, mu, logvar, args.beta, args.perceptual_weight)
                
                autoencoder_optimizer.zero_grad()
                total_loss.backward()
                autoencoder_optimizer.step()
                
                pbar.set_postfix(TotalLoss=total_loss.item(), ReconLoss=recon_loss.item(), KLLoss=kl_loss.item())
                logger.add_scalar("VAE/TotalLoss", total_loss.item(), global_step=epoch * l + i)
                logger.add_scalar("VAE/ReconstructionLoss", recon_loss.item(), global_step=epoch * l + i)
                logger.add_scalar("VAE/KLLoss", kl_loss.item(), global_step=epoch * l + i)
            
            # Обновляем learning rate для VAE
            autoencoder_scheduler.step()
            
            # Сохраняем VAE каждые 10 эпох
            if epoch % 10 == 0:
                torch.save(autoencoder.state_dict(), vae_checkpoint_path)
                logging.info(f"VAE saved at epoch {epoch}")
        
        # Сохраняем финальную версию VAE
        torch.save(autoencoder.state_dict(), vae_checkpoint_path)
        logging.info("VAE training completed and saved!")
    
    # Тестируем качество VAE
    logging.info("Testing VAE quality...")
    with torch.no_grad():
        test_images, _ = next(iter(dataloader))
        test_images = test_images[:4].to(device)  # Берем 4 изображения для теста
        
        recon_images, _, mu, logvar = autoencoder(test_images)
        test_loss, test_recon_loss, test_kl_loss = autoencoder.compute_loss(recon_images, test_images, mu, logvar, args.beta)
        
        logging.info(f"VAE Test - Total Loss: {test_loss.item():.4f}, "
                    f"Recon Loss: {test_recon_loss.item():.4f}, "
                    f"KL Loss: {test_kl_loss.item():.4f}")
    
    # Проверяем, есть ли уже обученная DiT модель
    dit_checkpoint_path = os.path.join("models", args.run_name, "dit_ckpt.pt")
    start_epoch = 0
    
    if os.path.exists(dit_checkpoint_path) and not args.retrain_dit:
        logging.info("Loading pre-trained DiT model...")
        model.load_state_dict(torch.load(dit_checkpoint_path, map_location=device))
        logging.info("DiT model loaded successfully!")
        start_epoch = args.epochs  # Пропускаем обучение, если модель уже обучена
    else:
        logging.info("Training Flow Matching model...")
    
    # Теперь обучаем Flow Matching модель
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        model.train()
        autoencoder.eval()  # Замораживаем автоэнкодер
        
        pbar = tqdm(dataloader)
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Кодируем изображения в латентное пространство
            with torch.no_grad():
                mu, logvar = autoencoder.encode(images)
                z1 = autoencoder.reparameterize(mu, logvar)
            
            # Создаем шум в латентном пространстве
            z0 = torch.randn_like(z1)
            
            # Иногда используем unconditional обучение для classifier-free guidance
            if args.conditional is False or np.random.random() < 0.1:
                labels = None
            
            # Вычисляем loss
            loss = flow_matching.compute_loss(model, z0, z1, labels)

            model_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            model_optimizer.step()

            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("FlowMatching/Loss", loss.item(), global_step=epoch * l + i)
            
            # Добавляем дополнительные метрики для диагностики
            with torch.no_grad():
                # Пересчитываем для логирования (или используем уже вычисленные значения)
                z_t, v_t, t = flow_matching.sample_flow_path(z0, z1)
                predicted_velocity = model(z_t, t, labels)
        
        # Обновляем learning rate для DiT
        model_scheduler.step()
        
        # Оценка модели и сохранение результатов каждые 10 эпох
        if epoch % 10 == 0:
            # Оценка качества модели
            logging.info("Evaluating model...")
            recon_loss, kl_loss, fm_loss = flow_matching.evaluate_model(model, autoencoder, dataloader, device)
            logger.add_scalar("Evaluation/ReconstructionLoss", recon_loss, global_step=epoch)
            logger.add_scalar("Evaluation/KLLoss", kl_loss, global_step=epoch)
            logger.add_scalar("Evaluation/FlowMatchingLoss", fm_loss, global_step=epoch)
            if args.conditional is True:
                labels = torch.arange(10).long().to(device)
                sampled_images = flow_matching.sample(model, autoencoder, n=len(labels), 
                                                   labels=labels, cfg_scale=3)
            else:
                sampled_images = flow_matching.sample(model, autoencoder, n=images.shape[0], 
                                                   labels=None, cfg_scale=0)
            
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            
            # Сохраняем модели
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"dit_ckpt.pt"))
            torch.save(autoencoder.state_dict(), os.path.join("models", args.run_name, f"vae_ckpt.pt"))


def launch_latent_flow_matching_dit():
    """Функция запуска обучения Latent Flow Matching + DiT"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Основные настройки
    args.run_name = "LatentFlowMatching_DiT"
    args.epochs = 200
    args.batch_size = 16
    args.image_size = 64
    args.dataset_path = r"E:\data_diffuse\datasets\cifar10-64\train"
    args.conditional = True
    args.num_classes = 10
    args.device = "cuda"
    args.lr = 2e-3  # Увеличено для более быстрого обучения
    args.autoencoder_lr = 2e-4
    args.weight_decay = 0.01
    
    # VAE настройки
    args.latent_size = 32
    args.latent_channels = 16  # Увеличено для лучшего качества
    args.base_channels = 64
    args.autoencoder_epochs = 150  # Больше эпох для лучшего обучения
    args.beta = 0.1  # Уменьшено для лучшего качества реконструкции
    args.perceptual_weight = 0.1  # Вес для perceptual loss
    args.retrain_vae = False  # Если True, переобучает VAE даже если есть сохраненные веса
    args.retrain_dit = False  # Если True, переобучает DiT даже если есть сохраненные веса
    
    # DiT настройки
    args.patch_size = 4
    args.embed_dim = 768  # Увеличено для лучшего качества
    args.depth = 16       # Больше слоев
    args.num_heads = 12   # Больше attention heads
    args.mlp_ratio = 4.0
    args.dropout = 0.05   # Уменьшено для лучшего обучения
    
    train_latent_flow_matching_dit(args)


def load_trained_models(run_name, device="cuda"):
    """Загружает обученные модели для генерации"""
    # Создаем улучшенный VAE
    autoencoder = ImprovedVAE(
        in_channels=3,
        latent_channels=16,
        base_channels=64
    ).to(device)
    
    # Создаем DiT модель
    model = LatentDiT(
        latent_size=32,
        latent_channels=16,  # Обновлено для соответствия новому VAE
        patch_size=4,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.1
    ).to(device)
    
    # Загружаем веса
    vae_path = os.path.join("models", run_name, "vae_ckpt.pt")
    dit_path = os.path.join("models", run_name, "dit_ckpt.pt")
    
    if os.path.exists(vae_path):
        autoencoder.load_state_dict(torch.load(vae_path, map_location=device))
        print(f"VAE loaded from {vae_path}")
    else:
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    
    if os.path.exists(dit_path):
        model.load_state_dict(torch.load(dit_path, map_location=device))
        print(f"DiT model loaded from {dit_path}")
    else:
        raise FileNotFoundError(f"DiT checkpoint not found at {dit_path}")
    
    return model, autoencoder


def generate_samples(run_name="LatentFlowMatching_DiT", num_samples=10, device="cuda"):
    """Генерирует образцы с помощью обученных моделей"""
    model, autoencoder = load_trained_models(run_name, device)
    
    # Создаем Flow Matching объект
    flow_matching = LatentFlowMatchingDiT(
        img_size=64,
        latent_size=32,
        latent_channels=16,  # Обновлено для соответствия новому VAE
        device=device
    )
    
    # Генерируем образцы
    with torch.no_grad():
        if num_samples <= 10:
            # Условная генерация для каждого класса
            labels = torch.arange(num_samples).long().to(device)
            sampled_images = flow_matching.sample(model, autoencoder, n=num_samples, 
                                               labels=labels, cfg_scale=3)
        else:
            # Безусловная генерация
            sampled_images = flow_matching.sample(model, autoencoder, n=num_samples, 
                                               labels=None, cfg_scale=0)
    
    # Сохраняем результат
    output_path = os.path.join("results", run_name, "generated_samples.jpg")
    save_images(sampled_images, output_path)
    print(f"Generated {num_samples} samples saved to {output_path}")
    
    return sampled_images


if __name__ == '__main__':
    launch_latent_flow_matching_dit()
