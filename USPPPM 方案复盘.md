# USPPPM 方案复盘

by Anntenna & fluentn & cty

## 赛题分析

### 背景

本次比赛的目标旨在通过从美国专利大型文本数据集中提取文字含义，帮助识别专利申请中的相似短语。专利检索和审查过程中，确定短语之间的语义相似性，对于确定一项发明是否曾被描述过至关重要。通俗点说，就是拿到一段专利申请中的文字描述，帮助判断该描述在它所申请的分类下，是否有相似的、已经申请了专利的条目；如果相似度大，那就说明已经存在该专利，初步判定结果是不该通过该申请；相似度小，说明该专利未被申请过，可以考虑通过。 <br>

在这个新型的语义相似性数据集上，通过匹配专利文档中的关键短语来提取相关信息、训练模型。这一挑战与标准语义相似性任务的不同之处在于，相似性需要在专利文本的所属的主题中进行评估，也就是其投递的专利上级分类（CPC classification)。<br>

### 任务

NLP语义相似度

### 数据示例

| id               | anchor               | target                    | context | score |
| ---------------- | -------------------- | ------------------------- | ------- | ----- |
| 37d61fd2272659b1 | abatement            | abatement of pollution    | A47     | 0.5   |
| 6bdd1d05ffa3401e | abatement            | emission abatement        | H04     | 0.5   |
| f6b53890ef57f9c5 | abnormal position    | unexpected position       | D03     | 0.75  |
| edd7a8b70dc94f43 | abnormal position    | hardware manufacturer     | E03     | 0     |
| 621b048d70aa8867 | absorbent properties | absorbent characteristics | D01     | 1     |

Context对应美国专利文本的分类，见titles.csv，information on the meaning of CPC codes may be found on the [USPTO website](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html). The CPC version 2021.05 can be found on the [CPC archive website](https://www.cooperativepatentclassification.org/Archive).

| code | title                                                        |
| ---- | ------------------------------------------------------------ |
| A    | HUMAN NECESSITIES                                            |
| A01  | AGRICULTURE; FORESTRY; ANIMAL HUSBANDRY; HUNTING; TRAPPING; FISHING |
| C    | CHEMISTRY; METALLURGY                                        |
| C01  | INORGANIC CHEMISTRY                                          |

## 总体思路

### 1  数据处理

- 数据清洗：（减小max_len，增大batch_size) 
  
  1. 大写 >>> 小写（有些大佬在论坛里说甚至要用小写的separation token [sep]）
  
     从下图可以看出，分词器并不能很好的识别大写的单词，将单词改成小写可以减少句子的最大长度。
  
     该方式将 max_len 从133降低到了107。
  
     <img src="D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220630133755927.png" alt="image-20220630133755927" style="zoom:67%;" />
  
  2.  乱码/分子式 >>> 删除
  
     还有一些极为个别的化学分子式，被识别成了各个数字以及乱码一样的东西。因为这部分内容实在不多，但是又结结实实的拉长了max_len，因此我们将下图所示的数据删除了，从而将max_len进一步降低到76。
  
     两步操作下来，max_len折减了近一半，大大节省了显存。
  
     <img src="D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220630134301879.png" alt="image-20220630134301879" style="zoom: 67%;" />
  
- 数据增强：（主要是增加随机扰动，因为我们目前不需要语序，都是一些词的堆砌，不需要让模型学一些奇怪的东西）
  1. random shuffle
  2. delete one word
  3. MLM：随机挑选一个词，替换为[MASK]

### 2  输入形式

1. **anchor + [SEP] + target + [SEP] + context**

   基本上是用这个接BERT+自行设计的下游结构

2. **anchor + [SEP] + context, target**

   主要用于[DebertaV2ForSequenceClassification](https://huggingface.co/transformers/v4.9.2/model_doc/deberta_v2.html#debertav2forsequenceclassification)

3. **anchor + [SEP] + context, target + [SEP] + context**

   主要用于双预训练模型的结构

### 3  模型

#### 3.1  预训练模型

##### 3.1.1  DeBERTa-v3-large

- **DeBERTa**

  1. **Disentangled positional and content information of individual tokens**

     BERT加入位置信息的方法是在输入embedding中加入postion embedding, pos embedding与char embeding和	segment embedding混在一起，这种早期就合并了位置信息在计算self-attention时，表达能力受限，维护信息非常被弱化了。

     ==本文的motivation就是将pos信息拆分出来，单独编码后去和content、和自己求attention==，增加计算 “位置-内容” 和 “内容-位置” 注意力的分散Disentangled Attention。为了更充分利用相对位置信息，输入的input embedding不再加入pos embeding, 而是input在经过transformer编码后，在encoder段与“decoder”段，通过**相对位置**计算**分散注意力**。

     <center class='half'>
         <img src='D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220629001748606.png' height = '300'/>
         <img src='D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220629003521977.png' height = '300'/>
     </center>
     
     
     
     </center>
     
     (Disentangled: 每一层都会加入一个不变的相对位置矩阵 $P$ )
     
     ==解码器前接入了绝对位置embedding，避免只有相对位置而丢失了绝对位置embeding==。
     
     (''_A new **store** opened beside a new **mall**._'' Using only the local context of **new** cannot help to predict 2 different words.)
     
     （其实本质就是在原始bert的倒数第二层transform中间层插入了一个分散注意力计算，而BERT是在最开始的时候放绝对位置的，DeBERTa原作怀疑一开始就放绝对位置会有损模型对相对位置的学习）
     $$
     \begin{aligned}
     A_{i, j} &=\left\{\boldsymbol{H}_{i}, \boldsymbol{P}_{i \mid j}\right\} \times\left\{\boldsymbol{H}_{j}, \boldsymbol{P}_{j \mid i}\right\}^{\top} \\
     &=\boldsymbol{H}_{i} \boldsymbol{H}_{j}^{\top}+\boldsymbol{H}_{i} \boldsymbol{P}_{j \mid i}^{\top}+\boldsymbol{P}_{i \mid j} \boldsymbol{H}_{j}^{\top}+\boldsymbol{P}_{i \mid j} \boldsymbol{P}_{j \mid i}^{\top}
     \end{aligned}
     $$
     即 $H_i$ 是内容编码，$P_{i \mid j}$ 是 $i$ 相对 $j$ 的位置编码（**相对位置**）， attention的计算中，融合了
     
     - 位置-位置，
     - 内容-内容，
     - 位置-内容 (I wanna exchange information with things at a position which is OO index away from me, and it would be dependent on what info they carry, instead of position)，
     - 内容-位置 (I am the verb "am", and I wanna attend to info around me >>> depends on position)。
     
     https://www.bilibili.com/video/BV1My4y1J7nh?spm_id_from=333.337.search-card.all.click&vd_source=4424849e6b7c34d7cfc7b22e5122a9b2
     $$
     \begin{aligned}
     &Q_{c}=H W_{q, c}, K_{c}=H W_{k, c}, V_{c}=H W_{v, c}, Q_{r}=P W_{q, r}, K_{r}=P W_{k, r}\\
     \end{aligned}
     $$
     
     $$
     &\tilde{A}_{i, j}=\underbrace{Q_{i}^{c} K_{j}^{c \top}}_{\text {(a) content-to-content }}+\underbrace{Q_{i}^{c} K_{\delta(i, j)}^{r}}_{\text {(b) content-to-position }}+\underbrace{K_{j}^{c} Q_{\delta(j, i)^{r}}^{\top}}_{\text {(c) position-to-content }}\\
     $$
     
     $$
     \boldsymbol{H}_{\boldsymbol{o}}=\operatorname{softmax}\left(\frac{\tilde{\boldsymbol{A}}}{\sqrt{3 d}}\right) \boldsymbol{V}_{c}
     $$
     
     其中 $Q_c$、$K_c$ 和 $V_c$ 分别是使用投影矩阵$W_{q, c}, W_{k, c}, W_{v, c} \in R^{d \times d}$ 生成的**投影内容向量**；$P \in R^{2 k \times d}$表示跨所有层共享的相对位置嵌入向量(即在正向传播期间保持不变)；$Q_r$ 和 $K_r$分别是使用投影矩阵 $W_{q, r}, \quad W_{k, r} \in R^{d \times d}$ 生成的**投影相对位置向量**。
     
     
     
     这里有一个问题，原作最后说，share the projection matrices of the relative position embedding ， $W_{q, r},  W_{k, r}$  with $W_{q, c},  W_{k, c}$ 如果用下图方式进行信息交换的话，实际上又回到了BERT那种最初始的方式了- -
     
     ![image-20220629005213762](D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220629005213762.png)
     
     https://zhuanlan.zhihu.com/p/344649850
     
  2. **相对位置的计算**
  
     限制了相对距离，相距大于一个阈值时距离就无效了，此时距离设定为一个常数，距离在有效范围内时，用参数来控制						
  
     $$
     \delta(i, j)=\left\{\begin{array}{rcl}
     0 & \text { for } & i-j \leqslant-k \\
     2 k-1 & \text { for } & i-j \geqslant k \\
     i-j+k & \text { others. } &
     \end{array}\right.
     $$
  
  3. **Decoding enhanced decoder**: Relative positional information in transformer, refeed the absolute postional information at the end
  
     **增强型解码器**（强行叫做解码器）
  
     用 EMD( enhanced mask decoder) 来代替原 BERT 的 SoftMax 层预测遮盖的 Token。
  
     因为我们在精调时一般会在 BERT 的输出后接一个特定任务的 Decoder，但是在预训练时却并没有这个 Decoder；所以本文在预训练时用一个两层的 Transformer decoder 和一个 SoftMax 作为 Decoder。**其实就是给后层的Transform encoder换了个名字，千万别以为是用到了Transformer的Decoder端。**
     
     
  
- **DeBERTaV3**: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing

  https://arxiv.org/abs/2111.09543

##### 3.1.2  Electra

Efficiently Learning an Encoder that Classifies Token Replacements Accurately

1. **NLP式的Generator-Discriminator**

   随机替换过于简单，作者借鉴GAN的思想，使用一个MLM的Generator-BERT来对输入句子进行更改，然后丢给Discriminator-BERT去判断哪个字被改过。

   输入句子经过生成器，输出改写过的句子，因为句子的字词是离散的，所以梯度在这里就断了，判别器的梯度无法传给生成器，于是生成器的训练目标还是MLM（作者在后文也验证了这种方法更好），判别器的目标是序列标注（判断每个token是真是假），两者同时训练，但**判别器的梯度不会传给生成器**

   ![img](https://pic4.zhimg.com/80/v2-2d8c42a08e37369b3b504ee006d169a3_720w.jpg)

2. **Replaced Token Detection**

   ELECTRA最主要的贡献是提出了新的预训练任务和框架，把生成式的Masked language model(MLM)预训练任务改成了判别式的Replaced token detection(RTD)任务，判断当前token是否被语言模型替换过。

   

https://zhuanlan.zhihu.com/p/89763176

​	MLM -> 判断某词是否错误 / 轻量

##### 3.1.3  BERT-for-patent

#### 3.2  预训练模型结构

1. 一个
2. 两个
3. Siamese Network
4. Fine-tuned DeBERTa Embedding + LGB

### 4  输出词向量

1. CLS

2. Last layer
3. mean(1st layer, last layer)
4. mean(倒数三层layer)

###  5  下游网络结构

1. Mean Pooling(只针对attention_type[i] == 1)
2. Sentence Classification
3. Attention
5. Light gbm

### 6  损失函数

- Pearson Loss
- BCE Loss
- MSE Loss

## 技巧尝试

Cross-Validation Ensemble (working)

几何平均 （not working not hurting)

白化 (not working)

对抗学习 / AWP (working)

## 经验总结

_**数据处理和数据增强真的很恨很恨很重要，折腾半天模型最多提升个0.01，数据上的改进可以轻轻松松提升0.02**_

1. 最后模型融合时，不要直接用求均值的办法，因为DeBERTa相对其他模型有压倒性优势，因此需要调高DeBERTa的比重
2. 有些大神在数据处理阶段，按照anchor和context进行分组，分组之后在同一组内的target全合并为target**s**（排除当前input自身的target），并拼接到每一个input之后（第一种输入形式+target**s**）
3. 预训练模型和自定义的下游网络结构可以使用不同的learning rate
4. CV也可以针对anchor而不是score进行分层采样；对于同一个网络架构，可以得到k个训练后的模型，根据OOF得分进行权重的选择

**其他需要进一步了解的操作**

1. 知识蒸馏
2. **?** Exponential Moving Average



