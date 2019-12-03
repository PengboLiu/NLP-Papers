### FREELB: ENHANCED ADVERSARIAL TRAINING FOR LANGUAGE UNDERSTANDING（ICLR 2020 under review）

优化目标如下：
$$
\min _{\boldsymbol{\theta}} \mathbb{E}_{(\boldsymbol{Z}, \boldsymbol{y}) \sim \mathcal{D}}\left[\frac{1}{K} \sum_{t=0}^{K-1} \max _{\boldsymbol{\delta}_{t} \in \mathcal{I}_{t}} L\left(f_{\boldsymbol{\theta}}\left(\boldsymbol{X}+\boldsymbol{\delta}_{t}\right), y\right)\right]
$$
相当于把原始的batch替换成了K倍的虚拟batch，一个虚拟的batch包括了$X+\delta_{0}, \dots, X+\delta_{K-1}$（都是一个一个的embedding）。基于PGD的对抗训练方法最小化每个训练样本的单个估计点的最大风险（有点绕）。而FreeLB方法可以几乎没有多余开销的情况下最小化每个梯度上升步骤的最大风险。

FreeLB方法的droput也区别于一般的fine tuning过程，在每个max步骤中，mask都是相同的。

FreeLB算法文字叙述如下：

每个epoch，每个minibatch内：初始化随机扰动 ${\delta}_{0}$ 和中间变量 $g_0$，然后从t = 1到K（K inner ascent steps）：迭代更新 ${\delta}_{t}$ 和 $g_t$，直到t = k时，梯度下降更新一次模型参数：$\theta \leftarrow \theta-\tau g_{K}​$。





### SMART （ICLR 2020 under review）

文章说被accept后会放出代码

主要是关于fine tuning的两点：

1、Smoothness-Inducing Adversarial Regularization ：看公式的意思是在损失函数中自己设计了一个正则项（文章叫smoothness-inducing adver- sarial regularizer）。**文章的意思是正则项使局部空间有连续性，所以即使改变一两个词输出也不会有变化**，这里阻止了Adversarial Attack的攻击。

2、Bregman Proximal Point Optimization ：在优化器做文章，似乎借用之前的方法（vanilla Bregman proximal point ）。优化参数时也有个正则项保证每次优化不会和上次的参数偏离太多。

3、Momentum的加速