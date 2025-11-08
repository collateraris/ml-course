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


class OptimalTransportFlowMatching:
    def __init__(self, img_size=256, device="cuda", sigma_min=1e-4, sigma_max=1.0, 
                 ot_reg=0.1, ot_max_iter=100, ot_tolerance=1e-6):
        """
        Optimal Transport Flow Matching модель для генерации изображений.
        
        Args:
            img_size: размер изображений
            device: устройство для вычислений
            sigma_min: минимальное значение шума
            sigma_max: максимальное значение шума
            ot_reg: параметр регуляризации для Sinkhorn алгоритма (entropy regularization)
            ot_max_iter: максимальное количество итераций Sinkhorn
            ot_tolerance: допустимая ошибка для сходимости Sinkhorn
        """
        self.img_size = img_size
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ot_reg = ot_reg
        self.ot_max_iter = ot_max_iter
        self.ot_tolerance = ot_tolerance

    def compute_cost_matrix(self, x0, x1):
        """
        Вычисляет матрицу стоимостей между x0 и x1.
        Использует L2 расстояние в пространстве изображений.
        
        Args:
            x0: начальные точки (шум) [B, C, H, W]
            x1: конечные точки (данные) [B, C, H, W]
            
        Returns:
            cost_matrix: матрица стоимостей [B, B]
        """
        # Преобразуем в векторы для вычисления расстояний
        x0_flat = x0.view(x0.shape[0], -1)  # [B, C*H*W]
        x1_flat = x1.view(x1.shape[0], -1)  # [B, C*H*W]
        
        # Вычисляем попарные L2 расстояния
        # cost[i, j] = ||x0[i] - x1[j]||^2
        x0_norm_sq = (x0_flat ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        x1_norm_sq = (x1_flat ** 2).sum(dim=1, keepdim=True).T  # [1, B]
        x0_x1 = torch.matmul(x0_flat, x1_flat.T)  # [B, B]
        
        cost_matrix = x0_norm_sq + x1_norm_sq - 2 * x0_x1
        return cost_matrix

    def sinkhorn_algorithm(self, cost_matrix, reg, max_iter=100, tolerance=1e-6):
        """
        Sinkhorn алгоритм для вычисления оптимального транспортного плана.
        
        Args:
            cost_matrix: матрица стоимостей [B, B]
            reg: параметр регуляризации
            max_iter: максимальное количество итераций
            tolerance: допустимая ошибка для сходимости
            
        Returns:
            transport_plan: оптимальный транспортный план [B, B]
        """
        B = cost_matrix.shape[0]
        
        # Единичные маргинальные распределения (uniform)
        a = torch.ones(B, device=self.device) / B  # источник
        b = torch.ones(B, device=self.device) / B  # назначение
        
        # Матрица ядра K = exp(-C / reg)
        # Для численной стабильности вычитаем минимум из каждой строки
        cost_matrix_normalized = cost_matrix - cost_matrix.min(dim=1, keepdim=True)[0]
        K = torch.exp(-cost_matrix_normalized / reg)
        
        # Инициализируем u
        u = torch.ones(B, device=self.device) / B
        
        for iteration in range(max_iter):
            u_prev = u.clone()
            
            # Обновляем v: v = b / (K.T @ u)
            KTu = K.T @ u
            v = b / (KTu + 1e-8)
            
            # Обновляем u: u = a / (K @ v)
            Kv = K @ v
            u = a / (Kv + 1e-8)
            
            # Проверяем сходимость
            if torch.norm(u - u_prev) < tolerance:
                break
        
        # Вычисляем оптимальный транспортный план: P = diag(u) @ K @ diag(v)
        transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        return transport_plan

    def sample_ot_coupling(self, x0, x1):
        """
        Сэмплирует пары (x0, x1) согласно оптимальному транспортному плану.
        
        Args:
            x0: начальные точки (шум) [B, C, H, W]
            x1: конечные точки (данные) [B, C, H, W]
            
        Returns:
            x0_matched: спаренные начальные точки [B, C, H, W]
            x1_matched: спаренные конечные точки [B, C, H, W]
        """
        # Вычисляем матрицу стоимостей
        cost_matrix = self.compute_cost_matrix(x0, x1)
        
        # Вычисляем оптимальный транспортный план
        transport_plan = self.sinkhorn_algorithm(
            cost_matrix, 
            reg=self.ot_reg,
            max_iter=self.ot_max_iter,
            tolerance=self.ot_tolerance
        )
        
        # Для оптимального спаривания используем жадное назначение
        # Берем argmax по каждому столбцу (для каждого x0 выбираем ближайший x1)
        # Но можно также использовать транспортный план напрямую
        
        # Вариант 1: Жадное назначение (быстрее)
        # Для каждого x0 выбираем x1 с наибольшей вероятностью в транспортном плане
        _, indices = torch.max(transport_plan, dim=1)
        
        # Вариант 2: Сэмплирование из распределения (более точно отражает OT план)
        # Нормализуем транспортный план в вероятности
        # prob_matrix = transport_plan / (transport_plan.sum(dim=1, keepdim=True) + 1e-8)
        # indices = torch.multinomial(prob_matrix, num_samples=1).squeeze(1)
        
        # Создаем спаренные образцы
        x0_matched = x0
        x1_matched = x1[indices]
        
        return x0_matched, x1_matched

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
        Генерирует изображения с помощью Optimal Transport Flow Matching.
        
        Args:
            model: обученная модель
            n: количество изображений для генерации
            labels: метки классов (для условной генерации)
            cfg_scale: масштаб classifier-free guidance
            num_steps: количество шагов для численного интегрирования
        """
        logging.info(f"Sampling {n} new images using Optimal Transport Flow Matching...")
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
        Вычисляет loss для Optimal Transport Flow Matching.
        
        Args:
            model: модель для предсказания вектора скорости
            x0: начальные точки (шум)
            x1: конечные точки (данные)
            labels: метки классов
        """
        # Используем оптимальный транспорт для спаривания образцов
        x0_matched, x1_matched = self.sample_ot_coupling(x0, x1)
        
        # Сэмплируем случайные моменты времени и пути
        x_t, v_t, t = self.sample_flow_path(x0_matched, x1_matched)
        
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
    
    # Создаем scheduler для learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
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
        
        # Обновляем learning rate после каждой эпохи
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("Learning_Rate", current_lr, global_step=epoch)
        
        # Сохраняем результаты каждые 10 эпох
        if epoch % 10 == 0:
            if args.conditional is True:
                labels = torch.arange(10).long().to(device)
                sampled_images = flow_matching.sample(ema_model, n=len(labels), labels=labels, cfg_scale=3)
            else:
                sampled_images = flow_matching.sample(model, n=images.shape[0], labels=None, cfg_scale=0)
            
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def train_optimal_transport_flow_matching(args):
    """Функция обучения Optimal Transport Flow Matching модели"""
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Используем ту же архитектуру UNet, что и в DDPM
    model = UNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Создаем scheduler для learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Создаем Optimal Transport Flow Matching объект
    ot_flow_matching = OptimalTransportFlowMatching(
        img_size=args.image_size, 
        device=device,
        ot_reg=getattr(args, 'ot_reg', 0.1),
        ot_max_iter=getattr(args, 'ot_max_iter', 100),
        ot_tolerance=getattr(args, 'ot_tolerance', 1e-6)
    )
    
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
            
            # Вычисляем loss с оптимальным транспортом
            loss = ot_flow_matching.compute_loss(model, noise, images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
        
        # Обновляем learning rate после каждой эпохи
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("Learning_Rate", current_lr, global_step=epoch)
        
        # Сохраняем результаты каждые 10 эпох
        if epoch % 10 == 0:
            if args.conditional is True:
                labels = torch.arange(args.num_classes).long().to(device)
                sampled_images = ot_flow_matching.sample(ema_model, n=len(labels), labels=labels, cfg_scale=3)
            else:
                sampled_images = ot_flow_matching.sample(model, n=images.shape[0], labels=None, cfg_scale=0)
            
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch_flow_matching():
    """Функция запуска обучения Flow Matching"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Настройки для условной генерации
    args.run_name = "FlowMatching_Conditional"
    args.epochs = 3000
    args.batch_size = 1
    args.image_size = 64
    args.dataset_path = r"E:\data_diffuse\datasets\cifar10-64-single1\train"
    args.conditional = False
    args.num_classes = 1
    args.device = "cuda"
    args.lr = 1e-5
    
    train_flow_matching(args)


def launch_optimal_transport_flow_matching():
    """Функция запуска обучения Optimal Transport Flow Matching"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Настройки для условной генерации
    args.run_name = "OptimalTransportFlowMatching_Conditional"
    args.epochs = 3000
    args.batch_size = 20
    args.image_size = 64
    args.dataset_path = r"E:\data_diffuse\datasets\cifar10-64-single1\train"
    args.conditional = False
    args.num_classes = 1
    args.device = "cuda"
    args.lr = 1e-5
    
    # Параметры оптимального транспорта
    args.ot_reg = 0.1  # Параметр регуляризации Sinkhorn
    args.ot_max_iter = 100  # Максимальное количество итераций Sinkhorn
    args.ot_tolerance = 1e-6  # Допустимая ошибка для сходимости
    
    train_optimal_transport_flow_matching(args)


if __name__ == '__main__':
    # Запуск обычного Flow Matching
    #launch_flow_matching()
    
    # Запуск Optimal Transport Flow Matching
    launch_optimal_transport_flow_matching()
