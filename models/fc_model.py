import torch 
import torch.nn as nn

class FCTextClassifyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(FCTextClassifyModel,self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        self.feature = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim*4),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim*4, num_classes),
        )

    def forward(self, x, mask=None):
        """
        :param x: [N, T] 
            LongTensor, 
            N:batch_size, T:seq_len,
            N个文本, 每个文本T个词的id
        :param mask: [N, T]
            数据的mask矩阵，如果样本对应token实际存在，那么值为1，否则为0
            在线推理的时候不需要填充或者批次大小为1，mask=None
        :return: [N, 2] 
            FloatTensor, 
            N:batch_size, 2:num_class
            N个文本, 每个文本对应的2个label的预测置信度
        """
        # 1. 单词特征提取
        z1 = self.embedding_layer(x)  # [N,T] → [N, T, embedding_dim]
        # 2. 文本特征提取
        z2 = self.feature(z1)  # [N, T, embedding_dim] → [N, T, embedding_dim*4]
        if mask is not None:
            mask = mask[..., None]  # [N,T] -> [N,T,1] 增加一个维度
            z2 = z2 * mask  # [N,T,4E] * [N,T,1] -> [N,T,4E]
            lengths = torch.sum(mask, dim=1)  # [N,T,1] -> [N,1]
            z3 = torch.sum(z2, dim=1) / lengths
        else:
            z3 = torch.mean(z2, dim=1)  # [N,T,4E] -> [N,4E]
        # 3. 文本特征池化
        # z3 = torch.mean(z2, dim=1)  # [N, T, embedding_dim*4] → [N, embedding_dim*4]
        # 4. 分类
        z4 = self.classifier(z3)  # [N, embedding_dim*4] → [N, 2]
        return z4