import torch
from torch.utils.data import Dataset, DataLoader


class LegalDataset(Dataset):
    """
    自定义 PyTorch Dataset，用于加载法律条款和对应的标签。

    Args:
        texts (list of str): 文本列表。
        labels (list of list[int]): 标签列表，每个标签是长度为 72 的二进制列表。
        tokenizer: 分词器实例。
        max_length (int): 最大序列长度。
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 校验所有标签长度为 72
        for label in labels:
            if len(label) != 72:
                raise ValueError(f"标签长度错误，应为 72，实际为: {len(label)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词编码
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),        # [seq_len]
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)      # [72] float 标签张量
        }


def create_data_loader(texts, labels, tokenizer, batch_size, max_length=256, shuffle=True):
    """
    创建 DataLoader。

    Args:
        texts (list of str): 文本列表。
        labels (list of list[int]): 标签列表。
        tokenizer: 分词器。
        batch_size (int): 批量大小。
        max_length (int): 最大序列长度。
        shuffle (bool): 是否打乱数据。

    Returns:
        DataLoader: PyTorch DataLoader 对象。
    """
    dataset = LegalDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
