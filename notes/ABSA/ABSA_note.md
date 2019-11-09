#### A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis

主要是Aspect-Guided Encoder，由两个GRU组成：Aspect-guided GRU 和 Transition GRU

- Aspect-guided GRU：初始阶段编码一些aspect相关和无关的信息；

- Transition GRU：每个时间步的T-GRU要作为输入，输入到下一个时间步的A-GRU；

Aspect-Reconstruction：为了把Aspect的信息编码到句子表示中，设计了重构方法。两个任务，两个损失函数。

#### Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks

引入了GCN

整个模型：

- Embedding 和 Bi-LSTM做底层表示；
- 句子的句法依存树作为GCN的输入得到面向aspect的特征，输出后做了一个mask。
- 上两步的输出做一个Aspect-aware Attention；
- softmax输出

感觉效果不是很好。

#### CAN: Constrained Attention Networks for Multi-Aspect Sentiment Analysis

特色在于 constrained attention networks。

ALSC （aspect level sentiment classification）Attention Layer：apsect和sentence的attention；

ACD（aspect category detection） Attention Layer：结构和ALSC一样的。

对应有两个prediction

整个模型加了一个Regularization Layer单元。

看表格结果比上一个paper要好

#### Syntax-Aware Aspect Level Sentiment Classification with Graph Attention Networks

TODO

