import re
import torch
from transformers import RobertaTokenizer

def clean_text(text):
    """
    清洗文本，移除不必要的字符和多余的空白。

    Args:
        text (str): 原始文本。

    Returns:
        str: 清洗后的文本。
    """
    # 移除 HTML 标签
    text = re.sub(r"<.*?>", "", text)

    # 替换多个空白为单个空格
    text = re.sub(r"\s+", " ", text)

    # 去除首尾空格
    text = text.strip()

    return text


def tokenize_texts(texts, tokenizer, max_length=256):
    """
    对文本列表进行分词并编码。

    Args:
        texts (list of str): 文本列表。
        tokenizer: 分词器实例（如 RobertaTokenizer）。
        max_length (int): 最大序列长度。

    Returns:
        dict: 编码后的输入，包括 input_ids 和 attention_mask。
    """
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded


def preprocess_data(data, tokenizer, max_length=256):
    """
    对数据集进行清洗、分词，并将标签转换为张量。

    Args:
        data (list of dict): 每条数据格式为 {'text': ..., 'label': [...] }。
        tokenizer: 分词器。
        max_length (int): 最大序列长度。

    Returns:
        list of dict: 每条样本包括 input_ids, attention_mask, label 的张量字典。
    """
    processed_data = []

    for item in data:
        # 清洗文本
        cleaned_text = clean_text(item["text"])

        # 分词和编码
        encoded = tokenize_texts([cleaned_text], tokenizer, max_length=max_length)

        # 确保标签为长度 72 的二进制张量
        label = item.get("label", [])
        if len(label) != 72:
            raise ValueError(f"Label length is not 72: {len(label)}. Please check the input data.")
        label_tensor = torch.tensor(label, dtype=torch.float)

        # 添加到预处理结果
        processed_data.append({
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": label_tensor
        })

    return processed_data


if __name__ == "__main__":
    # 示例数据
    raw_data = [
        {"text": "This is an example legal clause <b>with HTML</b>.", "label": [1] * 72},
        {"text": "Another clause, with unnecessary   spaces!", "label": [0, 1] * 36},
    ]

    # 加载分词器
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # 预处理数据
    processed_data = preprocess_data(raw_data, tokenizer)

    # 打印结果
    for item in processed_data:
        print(f"Input IDs: {item['input_ids'][:10]}...")  # 打印部分内容
        print(f"Attention Mask: {item['attention_mask'][:10]}...")
        print(f"Label: {item['label']}")
