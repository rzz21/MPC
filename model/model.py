import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

__all__ = ['MPDR']

r'''
Multipath Parameter Detection/Regression, LDAClassifier
'''

class ConvBNAC(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, pad=True, stride=1, groups=1):
        if pad:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(ConvBNAC, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True))
        ]))

class ResNet_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet_Block, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv_1', ConvBNAC(in_channel, out_channel, 3)),
            ('conv_2', ConvBNAC(out_channel, out_channel, 3))
        ]))
        self.path2 = ConvBNAC(in_channel, out_channel, 1)

    def forward(self, x):
        out = self.path1(x) + self.path2(x)

        return out
    
class MPDR(nn.Module):
    def __init__(self, c, w, h, param_num, path_Num):
        super(MPDR, self).__init__()
        self.path_Num = path_Num
        self.param_num = param_num
        self.model = nn.Sequential(OrderedDict([
            ('conv5x5', ConvBNAC(c, 20, 5)),
            ('res1', ResNet_Block(20, 20)),
            ('res2', ResNet_Block(20, 20)),
            ('res3', ResNet_Block(20, 40)),
            ('res4', ResNet_Block(40, 40)),
            # ('res5', ResNet_Block(40, 80)),
            # ('res6', ResNet_Block(80, 80)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(w * h * 0, 160)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('fc2', nn.Linear(160, param_num * path_Num)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True))
        ]))

    def forward(self, x):
        out = self.model(x)
        batch_num = x.size(0)
        out = out.view(batch_num, self.path_Num, self.param_num)
        
        return out

# class LDAClassifier(nn.Module):
#     def __init__(self, path_num, param_num, class_num):
#         super(LDAClassifier, self).__init__()
#         self.path_num = path_num
#         self.param_num = param_num
#         self.class_num = class_num
#         # Initialize LDA model
#         self.lda = LinearDiscriminantAnalysis(n_components=class_num - 1)

#     def forward(self, x, labels=None):
#         """
#         Forward pass for LDA classification.
#         Args:
#             x: Input numpy array of shape [batch_num, path_Num, param_num].
#             labels: Ground truth labels for supervised training (optional, numpy array).
#         Returns:
#             preds: Predicted class labels of shape [batch_num].
#         """
#         batch_num = x.shape[0]
#         # Reshape input to [batch_num, path_Num * param_num]
#         x_flat = x.reshape(batch_num, self.path_num * self.param_num)

#         if labels is not None:
#             # Fit LDA model if labels are provided (supervised training)
#             self.lda.fit(x_flat, labels)

#         # Predict class probabilities
#         preds = self.lda.predict(x_flat)
#         return preds
    

# class LDAClassifier(nn.Module):
#     def __init__(self, path_num, param_num, class_num):
#         """
#         增量式 LDA 分类器初始化
#         :param path_num: 每个样本的路径数量
#         :param param_num: 每条路径的参数数量
#         :param class_num: 分类的类别数量
#         """
#         super(LDAClassifier, self).__init__()
#         self.path_num = path_num
#         self.param_num = param_num
#         self.class_num = class_num

#         # 初始化统计量
#         self.total_samples = 0  # 总样本数
#         self.global_mean = None  # 全局均值
#         self.global_cov = None  # 全局协方差矩阵
#         self.class_means = {c: None for c in range(class_num)}  # 每个类别的均值
#         self.class_counts = {c: 0 for c in range(class_num)}  # 每个类别的样本数

#     def accumulate_batch(self, x, labels):
#         """
#         累积批次数据的统计量
#         :param x: 输入张量，形状为 [batch_size, path_num, param_num]
#         :param labels: 标签张量，形状为 [batch_size]
#         """
#         batch_size = x.size(0)
#         x_flat = x.view(batch_size, self.path_num * self.param_num).cpu().detach().numpy()
#         labels = labels.cpu().detach().numpy()

#         # 更新全局统计量
#         for c in range(self.class_num):
#             class_data = x_flat[labels == c]
#             if class_data.size == 0:
#                 continue

#             class_mean = np.mean(class_data, axis=0)
#             class_cov = np.cov(class_data, rowvar=False)

#             # 更新类别均值
#             if self.class_means[c] is None:
#                 self.class_means[c] = class_mean
#             else:
#                 self.class_means[c] = (self.class_means[c] * self.class_counts[c] + class_mean * class_data.shape[0]) / (self.class_counts[c] + class_data.shape[0])

#             # 更新类别样本数
#             self.class_counts[c] += class_data.shape[0]

#             # 更新全局协方差矩阵
#             if self.global_cov is None:
#                 self.global_cov = class_cov * class_data.shape[0]
#             else:
#                 self.global_cov += class_cov * class_data.shape[0]

#         # 更新全局样本数
#         self.total_samples += batch_size

#     def train_lda(self):
#         """
#         使用累积的统计量训练 LDA 模型
#         """
#         # 计算类间散度矩阵 (Sb)
#         overall_mean = np.mean([self.class_means[c] * self.class_counts[c] for c in range(self.class_num)], axis=0)
#         Sb = np.zeros((self.param_num * self.path_num, self.param_num * self.path_num))
#         for c in range(self.class_num):
#             if self.class_counts[c] > 0:
#                 diff = (self.class_means[c] - overall_mean).reshape(-1, 1)
#                 Sb += self.class_counts[c] * (diff @ diff.T)

#         # 计算类内散度矩阵 (Sw)
#         Sw = self.global_cov / self.total_samples

#         # 计算投影方向 (W)
#         eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
#         self.W = eigvecs[:, :self.class_num - 1].real  # 选择前 (class_num - 1) 个特征向量

#     def predict(self, x):
#         """
#         对输入数据进行预测
#         :param x: 输入张量，形状为 [batch_size, path_num, param_num]
#         :return: 预测的类别标签，形状为 [batch_size]
#         """
#         batch_size = x.size(0)
#         x_flat = x.view(batch_size, self.path_num * self.param_num).cpu().detach().numpy()
#         projected = x_flat @ self.W  # 投影到 LDA 空间
#         preds = np.argmax(projected, axis=1)  # 根据投影结果预测类别
#         return torch.tensor(preds, dtype=torch.long)
    
# # 参数设置
# batch_size = 200  # 每个批次的数据量
# num_batches = 50  # 总批次数量
# path_num = 10     # 每个样本的路径数量
# param_num = 5     # 每条路径的参数数量
# class_num = 3     # 分类的类别数量

# # 生成随机数据
# total_samples = batch_size * num_batches  # 总样本数量
# data = torch.randn(total_samples, path_num, param_num)  # 随机生成输入数据
# labels = torch.randint(0, class_num, (total_samples,))  # 随机生成标签

# # 初始化增量式 LDA 分类器
# lda_classifier = LDAClassifier(path_num, param_num, class_num)

# # 分批次累积数据
# for i in range(num_batches):
#     # 获取当前批次的数据和标签
#     start_idx = i * batch_size
#     end_idx = start_idx + batch_size
#     batch_data = data[start_idx:end_idx]
#     batch_labels = labels[start_idx:end_idx]

#     # 累积当前批次的数据
#     lda_classifier.accumulate_batch(batch_data, batch_labels)

# # 使用累积的数据训练 LDA 模型
# lda_classifier.train_lda()
# print("训练完成！")

# # 测试
# test_data = torch.randn(batch_size, path_num, param_num)  # 随机生成测试数据
# test_labels = torch.randint(0, class_num, (batch_size,))  # 随机生成测试标签

# # 预测
# predictions = lda_classifier.predict(test_data)
# print("预测结果:", predictions)

# # 计算准确率
# accuracy = (predictions == test_labels).float().mean()
# print(f"测试准确率: {accuracy.item() * 100:.2f}%")