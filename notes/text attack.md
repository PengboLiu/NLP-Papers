step1 单词重要程度排序

一句话中，只有某些单词是决定文本类别的。选择最能影响结果的单词。这里有一个重要性分数的概念；

step2 word transformer

满足三点：1、语义和原词相近	2、符合上下文	3、使模型的结果是错的

Synonym Extraction：使用embedding对整个词典求语义相似的词（word），50个相似度大于0.7的；

POS Checking： POS filter；

Semantic Similarity Checking：Universal Sentence Encoder把两个句子编码到高维空间然后计算相似度，大于阈值的加入到 FINCANDIDATES 中；

Finalization of Adversarial Examples：找语义最像或者预测分数最低的；

总的来说，step1是给单词重要性排序，step2是给句子中每个单词找一个替代



