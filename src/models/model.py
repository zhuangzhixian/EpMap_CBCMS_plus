from transformers import RobertaModel, RobertaConfig
import torch.nn as nn


class MultiLabelModel(nn.Module):
    def __init__(self, model_path="roberta/roberta-base", num_labels=72):
        """
        初始化基于 RoBERTa 的多标签分类模型，使用自定义分类头。

        Args:
            model_path (str): 本地或预训练的 RoBERTa 模型路径。
            num_labels (int): 标签数量。
        """
        super(MultiLabelModel, self).__init__()

        # 加载 RoBERTa 配置
        self.config = RobertaConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels

        # 加载 RoBERTa 主体模型（不含分类头）
        self.roberta = RobertaModel.from_pretrained(model_path, config=self.config)

        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # 自定义分类器，用于多标签输出
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        前向传播函数

        Args:
            input_ids (Tensor): 输入的 token id。
            attention_mask (Tensor): attention mask。

        Returns:
            Tensor: 多标签 logits 输出。
        """
        # 获取 RoBERTa 编码输出
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # 提取 [CLS] 位置的向量作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 经过 Dropout 和分类器
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits
