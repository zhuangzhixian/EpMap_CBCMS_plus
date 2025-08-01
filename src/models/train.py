import os
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, output_dir):
    """
    训练多标签分类模型。

    Args:
        model: PyTorch 模型。
        train_loader: 训练数据的 DataLoader。
        val_loader: 验证数据的 DataLoader。
        device: 设备 (CPU 或 GPU)。
        num_epochs: 训练轮数。
        learning_rate: 学习率。
        output_dir: 模型保存目录。
    """
    # 优化器：AdamW 是 Transformer 推荐优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度器：线性衰减
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 多标签任务使用 BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_labels = []
        total_preds = []

        # 遍历训练数据
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 前向传播
            logits = model(input_ids, attention_mask)

            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 收集预测值和真实标签
            preds = torch.sigmoid(logits).cpu().detach().numpy()
            total_preds.extend(preds)
            total_labels.extend(labels.cpu().numpy())

        # 输出训练损失与 F1 指标
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(
            total_labels, (np.array(total_preds) > 0.5).astype(int), average="macro"
        )
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | F1={f1:.4f}")

        # 验证
        validate_model(model, val_loader, device)

        # 保存模型权重
        save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.bin")
        torch.save(model.state_dict(), save_path)


def validate_model(model, val_loader, device):
    """
    在验证集上评估模型性能。

    Args:
        model: PyTorch 模型。
        val_loader: 验证数据的 DataLoader。
        device: 设备 (CPU 或 GPU)。
    """
    model.eval()
    total_labels = []
    total_preds = []

    with torch.no_grad():  # 验证不计算梯度
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 获取模型输出
            logits = model(input_ids, attention_mask)

            # 应用 sigmoid 并保存结果
            preds = torch.sigmoid(logits).cpu().detach().numpy()
            total_preds.extend(preds)
            total_labels.extend(labels.cpu().numpy())

    # 计算 F1 分数
    f1 = f1_score(
        total_labels, (np.array(total_preds) > 0.5).astype(int), average="macro"
    )
    print(f"Validation F1 Score: {f1:.4f}")
