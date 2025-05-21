# CNN_GAP4

数据集下载地址：
https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

本项目基于 PyTorch 实现了一种轻量级卷积神经网络（CNN），融合通道注意力机制、正则化分类器及多种图像增强与训练策略，在保持计算效率的前提下显著提升模型的特征建模能力、鲁棒性与泛化性能。

模型设计特点
1. 四层卷积结构：采用模块化设计的四层轻量级卷积网络，实现高效特征提取。
2. SE（Squeeze-and-Excitation）通道注意力机制：增强模型对关键信息通道的关注，提升表达能力。
3. 全局平均池化（GAP）结构：压缩特征图，减少参数，提高模型泛化能力。
4. 正则化分类器：通过 Dropout 与权重惩罚等策略降低过拟合风险。

图像增强策略（Albumentations）
引入 Albumentations 实现高效图像增强，包括：
1. 随机旋转与颜色扰动
2. Cutout 区域遮挡
3. 高斯噪声注入
目标：提升模型在真实场景下的鲁棒性与泛化能力。

模型训练与优化
1. 学习率调度器：使用 ReduceLROnPlateau 根据验证集表现动态调整学习率。
2. Early Stopping：当模型停止进步时提前终止训练，防止过拟合。

鲁棒性测试与可视化
针对三类典型扰动进行鲁棒性测试：
1. 高斯噪声扰动
2. 图像模糊（Blur）
3. 局部遮挡

使用 Grad-CAM 可视化工具 分析误分类样本的注意力热力图，观察模型注意力区域在干扰下的迁移变化与稳定性。

该项目的主要目的是学习卷积神经网络（CNN）的基础架构设计与实现。
所训练的模型参数保存在 .pth 文件中。在当前数据集上的最高准确率达到91.7%，训练尚未达到饱和状态，共进行了50轮训练。
训练过程中每轮的评估指标数据均保存于 .json 文件中，方便后续分析和对比。
整个模型的构建、训练流程、训练过程中的可视化分析以及模型的抗干扰能力测试均集成在 .ipynb 文件中，便于复现和深入理解。


Dataset download link:
https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

This project implements a lightweight convolutional neural network (CNN) based on PyTorch, integrating channel attention mechanisms, a regularized classifier, and various image augmentation and training strategies. It significantly enhances the model’s feature representation, robustness, and generalization performance while maintaining computational efficiency.

Model Design Features
1. Four-layer convolutional architecture: A modular design of a lightweight four-layer CNN for efficient feature extraction.
2. SE (Squeeze-and-Excitation) channel attention mechanism: Enhances the model’s focus on important feature channels to improve expressiveness.
3. Global Average Pooling (GAP): Compresses feature maps to reduce parameters and improve generalization.
4. Regularized classifier: Employs dropout and weight penalization techniques to reduce overfitting risk.

Image Augmentation Strategies (Albumentations)
Utilizes Albumentations for effective image augmentations, including:
1. Random rotations and color jitter
2. Cutout region masking
3. Gaussian noise injection
Objective: Improve the model’s robustness and generalization under real-world conditions.

Model Training and Optimization
Learning rate scheduler: Uses ReduceLROnPlateau to dynamically adjust the learning rate based on validation performance.
Early stopping: Terminates training early when no further improvement is observed to prevent overfitting.

Robustness Testing and Visualization
Robustness is evaluated under three typical perturbations:
1. Gaussian noise disturbance
2. Image blurring
3. Local occlusion

Grad-CAM visualization is used to analyze attention heatmaps of misclassified samples, observing how the model’s attention regions shift and remain stable under disturbances.

The primary goal of this project is to learn the foundational design and implementation of convolutional neural networks (CNNs).
The trained model parameters are saved in a .pth file. The highest accuracy achieved on the dataset is 91.7%, and the training has not yet reached saturation after 50 epochs.
Evaluation metrics for each training epoch are stored in a .json file for subsequent analysis and comparison.
The entire model construction, training process, visualization analysis, and robustness testing are integrated into a .ipynb notebook for easy reproduction and deeper understanding.



