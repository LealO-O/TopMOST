

import torch
from torch import nn
import torch.nn.functional as F
from .auto_diff_sinkhorn import sinkhorn_loss


class NSTM(nn.Module):
    '''
        Neural Topic Model via Optimal Transport. ICLR 2021

        He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.25, pretrained_WE=None, train_WE=True, embed_size=200, recon_loss_weight=0.07, sinkhorn_alpha=20):
        super().__init__()

        self.recon_loss_weight = recon_loss_weight
        self.sinkhorn_alpha = sinkhorn_alpha

        self.e1 = nn.Linear(vocab_size, en_units)
        self.e2 = nn.Linear(en_units, num_topics)
        self.e_dropout = nn.Dropout(dropout)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        # 预训练的词嵌入矩阵，如果它不是 None，则说明已经提供了预训练的词嵌入权重。
        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        #使用截断正态分布（truncated normal distribution）初始化主题嵌入矩阵 
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(self.topic_embeddings)

    def get_beta(self):  # 两个经过 L2 正则化的向量的模相乘的结果应该是它们的余弦相似度。
        word_embedding_norm = F.normalize(self.word_embeddings) # [vocab_size, embed_size]
        topic_embedding_norm = F.normalize(self.topic_embeddings) # [num_topics, embed_size]
        beta = torch.matmul(topic_embedding_norm, word_embedding_norm.T) # 矩阵乘法 [num_topics, vocab_size]
        return beta

    def get_theta(self, input): #在3.proposed model寻找相关
        theta = F.relu(self.e1(input)) # 通过 ReLU 激活函数，将所有负值置为0
        theta = self.e_dropout(theta) #进行 Dropout 操作,以防止过拟合。
        theta = self.mean_bn(self.e2(theta))#并通过 BatchNormalization 层 self.mean_bn，对结果进行标准化处理。
        theta = F.softmax(theta, dim=-1) # 最后，对标准化后的结果进行 Softmax 操作，得到文档的主题分布 theta，使得 theta 中每个元素都在 0 到 1 之间，并且所有元素的和为1，表示每个主题的概率。
        return theta  # 论文中的z = softmax(θ(~x))

    def forward(self, input):
        theta = self.get_theta(input)  #文档的主题分布  
        beta = self.get_beta()   #主题嵌入矩阵和词嵌入矩阵之间的相关性矩阵[num_topics, vocab_size]
        M = 1 - beta     #cost matrix
        sh_loss = sinkhorn_loss(M, theta.T, F.softmax(input, dim=-1).T, lambda_sh=self.sinkhorn_alpha)
        recon = F.softmax(torch.matmul(theta, beta), dim=-1)  #重构误差（定义为模型输出值与原始输入之间的均方误差）最小化
        recon_loss = -(input * recon.log()).sum(axis=1)

        loss = self.recon_loss_weight * recon_loss + sh_loss
        loss = loss.mean()
        return {'loss': loss}
