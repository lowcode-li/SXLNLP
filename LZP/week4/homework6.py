import torch
import math
import numpy as np
from transformers import BertModel



bert = BertModel.from_pretrained(r"D:\桌面\First grade\研一资料\nlp\录播\第六周 预训练模型\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()      #将BERT模型的所有参数（包括权重和偏置）保存在一个名为state_dict的字典中
bert.eval()                         #将BERT模型改成测试模式
#参数总计count
params  = 0
# 词汇表大小
vocab_size = 30522
# 隐藏单元大小
hidden_size = 768
#文本最长输入大小
max_position_embedding = 512

# 计算Embedding层参数量公式为:词向量参数+位置向量参数
#词向量参数为：vocab*hidden_size，位置向量参数：max_position_embedding*hidden_size
params_embedding = (vocab_size + max_position_embedding + 2) * hidden_size      #2是两个句子

print("BERT模型Embedding层的模型参数量为:", params_embedding)

#embedding层之后跟一层layer normalization层，参数包括均值和方差。
params_layer_normalization = hidden_size * 2

#Encoder层参数计算，包括多头注意力机制，前馈神经网络，以及残差归一化层
head_size = 12
params_multihead_attention = 3*hidden_size * (hidden_size * hidden_size/head_size)
params_feedforward_linear1 = hidden_size * hidden_size *4 * 2
params_feedforward_linear2 = hidden_size * hidden_size *4 + hidden_size
params_resnet = hidden_size * 2
params_encoder = 12 * (params_multihead_attention + params_feedforward_linear1 + params_feedforward_linear2+ params_resnet)

#pooler层
params_pooler = hidden_size * hidden_size + hidden_size

#总计参数
params = params_embedding + params_encoder + params_pooler
print(params)


num_params = sum(p.numel() for p in bert.parameters())

print(f"BERT模型的参数量为: {num_params}")

print(f'\n 理论计算：Bert参数量transformer: ', params)
