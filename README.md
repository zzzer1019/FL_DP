# FL_DP
## References
- H. Brendan McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, 2017, http://arxiv.org/abs/1602.05629.

- Martin Abadi et al., Deep Learning with Differential Privacy, 2016, https://arxiv.org/abs/1607.00133.

- Geyer R C, Klein T, Nabi M. Differentially private federated learning: A client level perspective[J]. arXiv preprint arXiv:1712.07557, 2017.

DATA：拆分完成的客户端数据
Dp：模型存档
MNIST_original：原始数据集
non_Dp：模型存档
Accountant.py：差分隐私辅助算法“时刻会计师”
Create_clients.py：模拟客户端数据
Federated_learning.py：联邦学习主体
Flask.py：网页端显示框架
Helper_Functions.py：联邦学习辅助函数，内包含差分隐私实现（高斯噪声）
mnist_inference.py：模型结构
MNIST_reader.py：读取数据集
randomized_response.py：随机响应机制（增强联邦学习客户端选择过程的随机性）
sample：主函数

-- Tensorflow==1.4.1 python2.7
