import copy
import os
import math

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class FlowMatching:
    def __init__(self, img_size=256, device="cuda", sigma_min=1e-4, sigma_max=1.0):
        """
        Flow Matching модель для генерации изображений.
        
        Args:
            img_size: размер изображений
            device: устройство для вычислений
            sigma_min: минимальное значение шума
            sigma_max: максимальное значение шума
        """
        self.img_size = img_size
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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

    def sample(self, model, n, labels=None, cfg_scale=0, num_steps=100):
        """
        Генерирует изображения с помощью Flow Matching.
        
        Args:
            model: обученная модель
            n: количество изображений для генерации
            labels: метки классов (для условной генерации)
            cfg_scale: масштаб classifier-free guidance
            num_steps: количество шагов для численного интегрирования
        """
        logging.info(f"Sampling {n} new images using Flow Matching...")
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
            model: модель для предсказания вектора скорости
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


def train_flow_matching(args):
    """Функция обучения Flow Matching модели"""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Используем ту же архитектуру UNet, что и в DDPM
    model = UNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Создаем Flow Matching объект
    flow_matching = FlowMatching(img_size=args.image_size, device=device)
    
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
        if epoch % 10 == 0:
            if args.conditional is True:
                labels = torch.arange(10).long().to(device)
                sampled_images = flow_matching.sample(ema_model, n=len(labels), labels=labels, cfg_scale=3)
            else:
                sampled_images = flow_matching.sample(model, n=images.shape[0], labels=None, cfg_scale=0)
            
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch_flow_matching():
    """Функция запуска обучения Flow Matching"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Настройки для условной генерации
    args.run_name = "FlowMatching_Conditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"E:\data_diffuse\datasets\cifar10-64\train"
    args.conditional = True
    args.num_classes = 10
    args.device = "cuda"
    args.lr = 3e-4
    
    train_flow_matching(args)


if __name__ == '__main__':
    launch_flow_matching()
