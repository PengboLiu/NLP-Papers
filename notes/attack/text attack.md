#### TextFooler

step1 单词重要程度排序

一句话中，只有某些单词是决定文本类别的。选择最能影响结果的单词。这里有一个重要性分数的概念；

step2 word transformer

满足三点：1、语义和原词相近	2、符合上下文	3、使模型的结果是错的

Synonym Extraction：使用embedding对整个词典求语义相似的词（word），50个相似度大于0.7的；

POS Checking： POS filter；

Semantic Similarity Checking：Universal Sentence Encoder把两个句子编码到高维空间然后计算相似度，大于阈值的加入到 FINCANDIDATES 中；

Finalization of Adversarial Examples：找语义最像或者预测分数最低的；

总的来说，step1是给单词重要性排序，step2是给句子中每个单词找一个替代。

#### Generating Natural Language Adversarial Examples



#### On the Robustness of Self-Attentive Models（ACL 2019）

contributions：1、novel algorithms生成对抗样本 2、基于self attention的模型，无论是预训练还是非预训练的，在对抗样本上的表现都比RNN要好 3、给出了相应的理论解释；

##### Attack方法（1-3是之前的方法，4和5是本文的工作）

1、Random Attack：顾名思义，在词表中随机找到一个词进行替换；

2、 List-based Attack：在近义词表中找到句子中每个词的替换，目的是让模型预测到错误的标签。近义词可以使用embedding计算相似度得到；

3、GS-GR：Greedy Select + Greedy Replace，名字就很粗暴，对一句话中每个词替换成padding，检测哪个词是”weak spot“，也就是最容易被攻击的。然后我们在词典中选择一个词去替换改词，直到模型预测错误标签为止。

4、Greedy Select + Embedding Constraint：GS-GR方法并没有考虑语义，可能把语句变成意义完全相反的句子。比如：“this is a good restaurant” 转变为“this is a bad restaurant.”。确实预测出错误标签，但是语义也完全改变了，这并不是我们的本意。作者的解决方法：使用句子级别的embedding作为约束。也就是说，替换了一个词，改动前后的sentence embedding不会变化太多。这种方法在方法三的基础上需要找到一个使改动前后的sentence embedding变化最小的单词；

5、Attention-based Select：基于attention的模型，改变attention score最高或最低的单词会影响模型的预测，根据这一点把对应的单词做替换。可以随机替换，文章称为AS-GR，也可以利用前面那种计算embedding相似度的方法，文章称之为AS-EC；