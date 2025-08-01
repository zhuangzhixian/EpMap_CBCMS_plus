import sys
import os
import pandas as pd
import torch
from transformers import RobertaTokenizer

# 添加 roberta 目录到模块搜索路径（适配本地加载）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
roberta_dir = os.path.join(base_dir, "roberta")
sys.path.extend([base_dir, roberta_dir])

# 导入自定义模块
from src.models.model import MultiLabelModel
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.data.dataset import create_data_loader


def load_data(file_path):
    """
    从 CSV 文件中加载训练数据，要求包含 'text' 和 'label' 两列。
    标签应为逗号分隔的二进制字符串。

    Args:
        file_path (str): CSV 文件路径。

    Returns:
        list[dict]: 每条数据为 {'text': str, 'label': List[int]}。
    """
    data = pd.read_csv(file_path)
    processed_data = []

    for _, row in data.iterrows():
        text = row["text"]
        label_str = row["label"]
        label = [int(x) for x in label_str.split(",")]
        if len(label) != 72:
            raise ValueError(f"标签长度应为 72，实际为: {len(label)}")
        processed_data.append({"text": text, "label": label})

    return processed_data


def main():
    # 配置参数
    config = {
        "data_dir": "data/",
        "output_dir": "output/",
        "max_length": 256,
        "batch_size": 32,
        "num_epochs": 1,
        "learning_rate": 5e-5,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_labels": 72,
        "model_path": "roberta/roberta-base",
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    # 加载数据
    train_data = load_data(os.path.join(config["data_dir"], "raw", "EpMap_train.csv"))
    val_data = load_data(os.path.join(config["data_dir"], "raw", "EpMap_validate.csv"))

    # 加载分词器和模型
    tokenizer = RobertaTokenizer.from_pretrained(config["model_path"])
    model = MultiLabelModel(
        model_path=config["model_path"],
        num_labels=config["num_labels"]
    ).to(config["device"])

    # 构建 DataLoader
    train_loader = create_data_loader(
        texts=[item["text"] for item in train_data],
        labels=[item["label"] for item in train_data],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    val_loader = create_data_loader(
        texts=[item["text"] for item in val_data],
        labels=[item["label"] for item in val_data],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    # 训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        output_dir=config["output_dir"]
    )

    # 评估（可选）
    print("\n=== Evaluating on Training Set ===")
    evaluate_model(model, train_loader, config["device"])

    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()
