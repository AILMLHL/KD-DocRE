import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class AFLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg):
        super().__init__()
        threshod = nn.Threshold(0, 0)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg


    def forward(self, logits, labels):
        # Adapted from Focal loss https://arxiv.org/abs/1708.02002, multi-label focal loss https://arxiv.org/abs/2009.14119
        # TH label 
        # AFL：自适应焦点损失，用于多标签分类中长尾类别。
        # 用与labels相同形状的零值张量初始化阈值类别（TH）。
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels) # to(labels) 确保 th_label 张量与 labels 张量位于同一设备上
        th_label[:, 0] = 1.0 # 将第一列（TH类别）设置为1。
        labels[:, 0] = 0.0 # 将实际标签的第一列重置为0。
        # 沿类别对标签求和，以找出具有多个标签的实体。
        label_idx = labels.sum(dim=1) # dim=1 表示沿着矩阵的行进行求和（即沿着类别的维度）。这个求和操作的结果是一个向量，

        # 根据标签计数识别不同类别的索引。
        two_idx = torch.where(label_idx==2)[0] 
        pos_idx = torch.where(label_idx>0)[0]

        neg_idx = torch.where(label_idx==0)[0]

        # 为正类和负类创建掩码。
        p_mask = labels + th_label # 正类掩码包括标签和TH类别。
        n_mask = 1 - labels # 负类掩码是标签的逆。
        neg_target = 1- p_mask # 负类目标。

        # 对标签张量进行形状推断。
        num_ex, num_class = labels.size() # 这里可以推断出labels是二维的，行代表样本数，列代表标签数
        num_ent = int(np.sqrt(num_ex)) # num_ent：实体数，从num_ex推断得出。
        # Rank each positive class to TH
        # 对每个类别的logits与阈值类别（TH）进行排名调整。
        logit1 = logits - neg_target * 1e30
        logit0 = logits - (1 - labels) * 1e30

        # Rank each class to threshold class TH
        # 为阈值类别创建掩码并与logits连接。
        th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1) # 对softmax归一化的对数概率。
        probs = torch.exp(F.log_softmax(logit_th, dim=1)) # 从对数概率得到概率。

        # Probability of relation class to be positive (1)
        # 计算成为正类（1）和负类（0）的概率。
        prob_1 = probs[:, 0 ,:] # 每个类别成为正类的概率。
        # Probability of relation class to be negative (0)
        prob_0 = probs[:, 1 ,:] # 每个类别成为负类的概率。
        prob_1_gamma = torch.pow(prob_1, self.gamma_neg) # 使用gamma调整的负类概率。
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos) # 使用gamma调整的正类概率。
        log_prob_1 = log_probs[:, 0 ,:] # 正类的对数概率。
        log_prob_0 = log_probs[:, 1 ,:] # 负类的对数概率。
        
        
        


        # Rank TH to negative classes
        # 对阈值类别（TH）与负类进行排名
        logit2 = logits - (1 - n_mask) * 1e30
        # 通过为非负类设置高负值来调整logits。
        # 这有助于将阈值类别（TH）与负类进行排名。

        # 对调整后的logits应用log softmax进行归一化。
        # 这计算了每个类别成为阈值类的对数概率。
        rank2 = F.log_softmax(logit2, dim=-1)

        # 计算针对正类的损失的第一部分。
        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
        # 这部分损失关注于正类。
        # 它使用成为正类的对数概率（log_prob_1），
        # 并通过成为负类的调整概率（prob_0_gamma）进行缩放。
        # 损失通过实际标签加权，以关注相关的正类。

        # 计算针对阈值类的损失的第二部分。
        loss2 = -(rank2 * th_label).sum(1) 
        # 这部分损失关注于阈值类别（TH）。
        # 它将归一化的logits（rank2）与阈值类别标签相乘。
        # 跨类别求和使损失集中在阈值类别上。

        
        # 结合两部分损失。
        loss =  1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
        # 最终损失是loss1和loss2的加权和。
        # loss1在类别之间求和，并在样本之间求平均。
        # loss2直接在样本之间求平均。
        # 在最终损失计算中，这两部分被等权重处理。

        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1) * 1.0
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    
