import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import numpy as np


def evaluate_model(model, data_loader, device):
    """
    在测试集上评估模型性能，返回主要指标。

    Args:
        model: 已训练好的 PyTorch 模型。
        data_loader: 测试数据的 DataLoader。
        device: 使用的设备（CPU 或 GPU）。

    Returns:
        metrics (dict): 包含 F1 分数、Precision、Recall 的评估结果。
    """
    model.eval()  # 设置为评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 不计算梯度
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 校验标签维度
            if labels.size(1) != 72:
                raise ValueError(f"标签维度错误，应为72，实际为: {labels.size(1)}")

            # 前向传播，获取 logits
            logits = model(input_ids, attention_mask)

            # 应用 Sigmoid 进行多标签概率化预测
            preds = torch.sigmoid(logits).cpu().numpy()

            # 收集预测值与真实标签
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 概率转为二进制标签（0/1）
    all_preds = (np.array(all_preds) > 0.5).astype(int)

    # 计算评估指标（加权平均）
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    # 输出结果字典
    metrics = {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
    }

    print("评估指标：")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def generate_classification_report(model, data_loader, device, label_names=None):
    """
    生成详细的分类报告（每个标签的 Precision / Recall / F1）。

    Args:
        model: 已训练好的 PyTorch 模型。
        data_loader: 测试数据的 DataLoader。
        device: 使用的设备（CPU 或 GPU）。
        label_names: 可选参数，标签名称列表（用于报告显示）。

    Returns:
        report (str): 文本形式的分类报告。
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 校验标签维度
            if labels.size(1) != 72:
                raise ValueError(f"标签维度错误，应为72，实际为: {labels.size(1)}")

            # 前向传播获取 logits
            logits = model(input_ids, attention_mask)

            # 概率输出
            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 转换为二值标签
    all_preds = (np.array(all_preds) > 0.5).astype(int)

    # 生成详细分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0  # 避免除0警告
    )

    print("分类报告：")
    print(report)

    return report
