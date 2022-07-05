# U.S.-Patent-Phrase-to-Phrase-Matching
Kaggle NLP Competition, 2022

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

- **数据清洗**：（减小max_len，增大batch_size) 

  1. **大写 >>> 小写**（有些大佬在论坛里说甚至要用小写的separation token [sep]）

     从下图可以看出，分词器并不能很好的识别大写的单词，将单词改成小写可以减少句子的最大长度。

     该方式将 max_len 从133降低到了107。

     <a href="https://sm.ms/image/YIsg91qahDOoLjr" target="_blank"><img src="https://s2.loli.net/2022/07/05/YIsg91qahDOoLjr.png" width=400></a>

  2. **乱码/分子式 >>> 删除**

     还有一些极为个别的化学分子式，被识别成了各个数字以及乱码一样的东西。因为这部分内容实在不多，但是又结结实实的拉长了max_len，因此我们将下图所示的数据删除了，从而将max_len进一步降低到76。

     两步操作下来，max_len折减了近一半，大大节省了显存。

     <a href="https://sm.ms/image/uMDFmB2Rg1KO6Gz" target="_blank"><img src="https://s2.loli.net/2022/07/05/uMDFmB2Rg1KO6Gz.png" width=600></a>

- **数据增强**：（主要是增加随机扰动，因为我们目前不需要语序，都是一些词的堆砌，不需要让模型学一些奇怪的东西）

  1. random shuffle
  2. delete one word
  3. MLM：随机挑选一个词，替换为[MASK]

### 2  输入形式

一段式和二段式主要影响输入的segment embedding，从分词器出来的向量中，token_type_ids中的元素会成为 [0,0,...,1,1,...,0,...,0] 的形式，前面的0表示第一段话，中间的1表示第二段话，最后的0表示padding。目前抱抱脸上的模型只支持最多两段，如果需要更多segments，可以自己训练。

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

  > 1. **Disentangled positional and content information of individual tokens**
  >
  >    BERT加入位置信息的方法是在输入embedding中加入postion embedding, pos embedding与char embeding和segment embedding混在一起，这种早期就合并了位置信息在计算self-attention时，表达能力受限，维护信息非常被弱化了。
  >
  >    ==本文的motivation就是将pos信息拆分出来，单独编码后去和content、和自己求attention==，增加计算 “位置-内容” 和 “内容-位置” 注意力的分散Disentangled Attention。为了更充分利用相对位置信息，输入的input embedding不再加入pos embeding, 而是input在经过transformer编码后，在encoder段与“decoder”段，通过**相对位置**计算**分散注意力**。
  >
  > 
  >
  >    (Disentangled: 每一层都会加入一个不变的相对位置矩阵 $P$ )
  >
  >    ==解码器前接入了绝对位置embedding，避免只有相对位置而丢失了绝对位置embeding==。
  >
  >    （其实本质就是在原始bert的倒数第二层transform中间层插入了一个分散注意力计算，而BERT是在最开始的时候放绝对位置的，DeBERTa原作怀疑一开始就放绝对位置会有损模型对相对位置的学习）
  > $$
  >    \begin{aligned}
  >    A_{i, j} &=\left\{\boldsymbol{H}_{i}, \boldsymbol{P}_{i \mid j}\right\} \times\left\{\boldsymbol{H}_{j}, \boldsymbol{P}_{j \mid i}\right\}^{\top} \\
  >    &=\boldsymbol{H}_{i} \boldsymbol{H}_{j}^{\top}+\boldsymbol{H}_{i} \boldsymbol{P}_{j \mid i}^{\top}+\boldsymbol{P}_{i \mid j} \boldsymbol{H}_{j}^{\top}+\boldsymbol{P}_{i \mid j} \boldsymbol{P}_{j \mid i}^{\top}
  >    \end{aligned}
  > $$
  >    即 $H_i$ 是内容编码，$P_{i \mid j}$ 是 $i$ 相对 $j$ 的位置编码（**相对位置**）。
  >
  > $$
  >    \begin{aligned}
  >    &Q_{c}=H W_{q, c}, K_{c}=H W_{k, c}, V_{c}=H W_{v, c}, Q_{r}=P W_{q, r}, K_{r}=P W_{k, r}\\
  >    \end{aligned}
  > $$
  >
  > $$
  >    &\tilde{A}_{i, j}=\underbrace{Q_{i}^{c} K_{j}^{c \top}}_{\text {(a) content-to-content }}+\underbrace{Q_{i}^{c} K_{\delta(i, j)}^{r}}_{\text {(b) content-to-position }}+\underbrace{K_{j}^{c} Q_{\delta(j, i)^{r}}^{\top}}_{\text {(c) position-to-content }}\\
  > $$
  >
  > $$
  >    \boldsymbol{H}_{\boldsymbol{o}}=\operatorname{softmax}\left(\frac{\tilde{\boldsymbol{A}}}{\sqrt{3 d}}\right) \boldsymbol{V}_{c}
  > $$
  >
  >    其中 $Q_c$、$K_c$ 和 $V_c$ 分别是使用投影矩阵$W_{q, c}, W_{k, c}, W_{v, c} \in R^{d \times d}$ 生成的**投影内容向量**；$P \in R^{2 k \times d}$表示跨所有层共享的相对位置嵌入向量(即在正向传播期间保持不变)；$Q_r$ 和 $K_r$分别是使用投影矩阵 $W_{q, r}, \quad W_{k, r} \in R^{d \times d}$ 生成的**投影相对位置向量**。
  >
  >    https://zhuanlan.zhihu.com/p/344649850
  >
  >    https://www.bilibili.com/video/BV1My4y1J7nh?spm_id_from=333.337.search-card.all.click&vd_source=4424849e6b7c34d7cfc7b22e5122a9b2
  >
  > 2. **相对位置的计算**
  >
  >    限制了相对距离，相距大于一个阈值时距离就无效了，此时距离设定为一个常数，距离在有效范围内时，用参数来控制						
  >
  >    $$
  >    \delta(i, j)=\left\{\begin{array}{rcl}
  >    0 & \text { for } & i-j \leqslant-k \\
  >    2 k-1 & \text { for } & i-j \geqslant k \\
  >    i-j+k & \text { others. } &
  >    \end{array}\right.
  >    $$
  >
  > 3. **Decoding enhanced decoder**: Relative positional information in transformer, refeed the absolute postional information at the end
  >
  >    **增强型解码器**（强行叫做解码器）
  >
  >    用 EMD( enhanced mask decoder) 来代替原 BERT 的 SoftMax 层预测遮盖的 Token。
  >
  >    因为我们在精调时一般会在 BERT 的输出后接一个特定任务的 Decoder，但是在预训练时却并没有这个 Decoder；所以本文在预训练时用一个两层的 Transformer decoder 和一个 SoftMax 作为 Decoder。**其实就是给后层的Transform encoder换了个名字，千万别以为是用到了Transformer的Decoder端。**
  >
  > 

- **DeBERTaV3**: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing

  原作论文 https://arxiv.org/abs/2111.09543

  1. 预训练任务从mask language modeling(MLM)换成了replaced token detection(RTD)从而更有效的利用样本在预训练任务中(本质是一种Electra-style的预训练任务)

  2. 使用了一种gradient-disentangled embedding sharing方法，使模型能够实现embedding sharing 同时避免了'tug-of-war'问题（避免了discriminator 和 generator 拔河）

##### 3.1.2  Electra

Efficiently Learning an Encoder that Classifies Token Replacements Accurately

> 1. **NLP式的Generator-Discriminator**
>
>    随机替换过于简单，作者借鉴GAN的思想，使用一个MLM的Generator-BERT来对输入句子进行更改，然后丢给Discriminator-BERT去判断哪个字被改过。
>
>    <img src="https://pic4.zhimg.com/80/v2-2d8c42a08e37369b3b504ee006d169a3_720w.jpg" alt="img" style="zoom: 67%;" />
>
>    目标函数：$\min _{\theta_{G}, \theta_{D}} \sum_{x \in \mathcal{X}} \mathcal{L}_{M L M}\left(x, \theta_{G}\right)+\lambda \mathcal{L}_{D i s c}\left(x, \theta_{D}\right)$
>
>    小的generator以及discriminator的方式共同训练，并且采用了两者loss相加，使得discriminator的学习难度逐渐地提升，学习到更难的token（plausible tokens）。
>
> 2. **Replaced Token Detection**
>
>    ELECTRA最主要的贡献是提出了新的预训练任务和框架，把生成式的Masked language model(MLM)预训练任务改成了判别式的Replaced token detection(RTD)任务，判断当前token是否被语言模型替换过。
>
>    输入句子经过生成器，输出改写过的句子，因为句子的字词是离散的，所以梯度在这里就断了，判别器的梯度无法传给生成器（generator的sampling的步骤导致），于是生成器的训练目标还是MLM（作者在后文也验证了这种方法更好），判别器的目标是序列标注（判断每个token是真是假），两者同时训练，但**判别器的梯度不会传给生成器**。
>
> 3. 其他训练点
>
>    - 作者设置了相同大小的生成器和判别器，训练D时不会再更新G的参数，在不共享权重下的效果是83.6，只共享token embedding层 （不包括encoder，只有token和postion embedding的参数）的效果是84.3，共享所有权重的效果是84.4，后面两种情况相差不大。作者认为**生成器对embedding有更好的学习能力**，因为在计算MLM时，softmax是建立在所有vocab上的，之后反向传播时会更新所有embedding，而判别器只会更新输入的token embedding。最后作者只使用了embedding sharing。
>    - 作者在保持原有hidden size的设置下减少了生成器层数（保持embedding，减少encoder），**生成器的大小在判别器的1/4到1/2之间效果是最好的**。作者认为原因是过强的生成器会增大判别器的难度。
>
> https://zhuanlan.zhihu.com/p/89763176
>
> https://zhuanlan.zhihu.com/p/118135466

##### 3.1.3  BERT-for-patent

理解成在专利语料上训练的BERT就可以



#### 3.2  预训练模型结构

1. 一个BERT

   <img src="D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220705155332779.png" alt="image-20220705155332779" style="zoom: 67%;" />

2. 两个BERT

   <img src="D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220705155354513.png" alt="image-20220705155354513" style="zoom:67%;" />

3. Siamese Network (两个BERT，但是共享权重)

   <img src="D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\image-20220705155408129.png" alt="image-20220705155408129" style="zoom:67%;" />

4. Fine-tuned DeBERTa Embedding + LGB （将BERT的输出作为LightGBM的输入）

### 4  输出词向量

1. CLS

2. Last layer
3. mean(1st layer, last layer)
4. mean(倒数三层layer)

###  5  下游网络结构

1. Mean Pooling(只针对attention_type[i] == 1)

   ~~~python
   outputs = self.model(**inputs)
   last_hidden_state = outputs[0] # (batch_size, max_length, 1024)
   feature = torch.mean(last_hidden_states, 1) # (batch_size, 1024)
   ~~~

2. Sentence Classification

   [DebertaV2ForSequenceClassification](https://huggingface.co/transformers/v4.9.2/model_doc/deberta_v2.html#debertav2forsequenceclassification)

3. Attention (相当于设计了加权平均，weights会随着last_hidden_states最终找到一个稳定值)

   ```python
   # model
   self.attention = nn.Sequential(
       nn.Linear(self.config.hidden_size, 512),
       nn.Tanh(),
       nn.Linear(512, 1),
       nn.Softmax(dim=1)
   )
   
   # forward
   outputs = self.model(**inputs)
   last_hidden_state = outputs[0] # (batch_size, max_length, 1024)
   weights = self.attention(last_hidden_states)
   feature = torch.sum(weights * last_hidden_states, dim=1) # (batch_size, 1024)
   
   ```

4. Light gbm

### 6  损失函数

- Pearson Loss （直接用Pearson系数作为loss）
  $$
  \boldsymbol{r}_{x y}=\frac{\sum((x-\bar{x})(y-\bar{y}))}{\sqrt{\sum(x-\bar{x})^{2} \sum(y-\bar{y})^{2}}}
  $$

  ~~~python
  def pearsonr(x, y):
      """
      Mimics `scipy.stats.pearsonr`
      Arguments
      ---------
      x : 1D torch.Tensor
      y : 1D torch.Tensor
      Returns
      -------
      r_val : float
          pearsonr correlation coefficient between x and y
      
      Scipy docs ref:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
      
      Scipy code ref:
          https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
      Example:
          >>> x = np.random.randn(100)
          >>> y = np.random.randn(100)
          >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
          >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
          >>> np.allclose(sp_corr, th_corr)
      """
      mean_x = torch.mean(x)
      mean_y = torch.mean(y)
      xm = x.sub(mean_x)
      ym = y.sub(mean_y)
      r_num = xm.dot(ym)
      r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
      r_val = r_num / r_den
      return r_val
  ~~~

- BCE Loss

  ~~~python
  criterion = nn.BCEWithLogitsLoss(reduction="mean")
  loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
  ~~~

  - **BCEWithLogitsLoss** (我们用的)

    Pytorch 官方解释： https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    本质是BCELoss和Sigmoid合成一步计算，解决了log-sum-exp的问题

    > **调用过程**
    >
    > Inpt: tensor([[ 0.5744],  [-1.7654],  [-0.0874]])  （比如说模型的output是batchsize * 1）
    >
    > Label: tensor([[1.0000],  [0.2500],  [0.5000]])
    >
    > 
    >
    > 以下这两种计算方式是等价的：
    >
    > 1. BCEWithLogitsLoss()
    >
    >    ~~~python
    >    criterion = nn.BCEWithLogitsLoss()
    >    criterion(inpt, label)
    >    ~~~
    >
    > 2. Sigmoid之后算Cross Entropy
    >
    >    ~~~python
    >    torch.sum(inpt.sigmoid().log()*target + (1-target)*(1-inpt.sigmoid()).log())/3
    >    ~~~
    >
    >    ![未命名图片](D:\Personal Files\学习类\Project\Kaggle\U.S. Patent Phrase to Phrase Matching\USPPPM 方案复盘.assets\未命名图片.png)
    >
    > 

- MSE Loss
  $$
  M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\widehat{y}_{i}\right)^{2}
  $$

## 技巧尝试

- **Cross-Validation** (working)

  | 方案             | 描述                                                         |
  | ---------------- | ------------------------------------------------------------ |
  | Cross-Validation | 分层策略：不仅可以根据label(score)分类分层交叉验证，还可以根据某种feature(anchor)分层 |
  | CV self-ensemble | 通常使用 5-fold CV 的self-ensemble (但有人表示15-fold的结果更好） |
  | Nested CV        | 可以预留20%的数据作为test集                                  |

- **Ensemble** (working)

  | 方案                          | 描述                                                         |
  | ----------------------------- | ------------------------------------------------------------ |
  | k-fold  self-ensemble         | k可以取 5 10 15                                              |
  | 5-folds  + 1 full train model | Full train的Weight是5-folds的两倍                            |
  | Weighted ensemble             | DeBERTa的效果显著高于其他模型，理应拥有更高的占比            |
  | weight调试方法                | 用out-of-fold测试不同模型的占比，可以用linear regression直接算 |

- **几何平均** （not working not hurting)

  在Ensemble这一步，针对多个模型的预测结果，做几何平均，而不是简单投票或者加权平均。

- **白化** (not working)

  主要针对双BERT的模型结构。白化方法主要想解决句子 embedding 的各向异性及向量的分布不均匀问题。在训练数据中计算出来参数，然后在测试数据中使用这个参数直接去做转化模型得到的hidden layer，用白化后的向量去算cosine similarity。

  > **BERT存在的两个问题**
  >
  > 1. BERT encode出来的向量表达具有各向异性
  >
  >    什么叫各向异性？举个例子，一些电阻原件，正接是良导体，反接是绝缘体或者电阻很大，沿不同方向差异很大。
  >
  >    在 BERT 出来的向量中表现为，用不同的方式去衡量它，他表现出不同的语义，差别很大，也就是不能完整的衡量出 BERT 向量中全部语义信息。各向异性的表现状态，就是向量会不均匀分布，且充斥在一个狭窄的锥形空间下。
  >
  >    这种性质也限制了句子向量的语义表达能力，因此当采用计算 BERT encode 句子相似度，采用 cos 或 dot 是无法很好的衡量出两个句子的相似度的，因为 BERT 向量不是基于一个标准正交基得到的。
  >
  > 2. 分布不均匀：低频词稀疏，高频词紧密
  >
  >    高频词会集中在头部，离原点近，低频词会集中在尾部，离远点远高频词与低频词分布在不同的区域，那高频词与低频词之间的相似度也就没法计算了。这也反映出来的就是明显的低频词没有得到一个很好的训练。同时，高频词频次高，也会主宰句子表达。
  >
  >    <img src="https://img-blog.csdnimg.cn/img_convert/b93cc1c528eef25bdf713c94d33a653d.png" alt="b93cc1c528eef25bdf713c94d33a653d.png" style="zoom:50%;" />
  >
  >    BERT-whitening 提出通过一个白化的操作直接校正局向量的协方差矩阵，$\widetilde{x}_{i}=\left(x_{i}-\mu\right) W$，问题就变成计算 BERT 向量分布的均值以及协方差。
  >
  > 
  >
  > [文本表达进击：从BERT-flow到BERT-whitening、SimCSE](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/121259650?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121259650-blog-120122617.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121259650-blog-120122617.pc_relevant_antiscanv2&utm_relevant_index=10)
  >
  > [无监督语义匹配之BERT-Whitening](https://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247485752&idx=1&sn=5352ef399885fec801779069e9fa780c&chksm=e87d3b1edf0ab208768400b0c01dcf6141896246caea9de1d9141822f477a48ae561d9897ac2&token=1045652591&lang=zh_CN#rd)

  

- **对抗权重扰动 / AWP / Adversarial Weight Perturbation**  (working)

  大量实验表明，对抗学习后的模型泛化能力变强了。因此在本次NLP任务中，我们使用作为一种regularization的方式，在embedding上做扰动，尝试提高模型的泛化能力。


  > **对抗训练基本思想：Min-Max公式**
  > $$
  > \min _{\theta} \mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\max _{r_{a d v} \in \mathcal{S}} L\left(\theta, x+r_{a d v}, y\right)\right]
  > $$
  > 中括号里的含义为我们要找到一组在样本空间内、使Loss最大的的对抗样本（该对抗样本由原样本x和经过某种手段得到的扰动项r_adv共同组合得到）。这样一组样本组成的对抗样本集，它们所体现出的数据分布，就是该中括号中所体现的。
  > 外层min()函数指的则是，我们面对这种数据分布的样本集，要通过对模型参数的更新，使模型在该对抗样本集上的期望loss最小。
  >
  > **Motivation**
  >
  > 用被对抗性样本污染过的训练样本来训练模型，直到模型能学习到如此类型的抵抗，从而保证模型的安全性。
  >
  > **如何找到最佳扰动r_adv呢？**
  >
  > 首先先做梯度上升（**梯度直接用于更新函数**），找到最佳扰动r，使得loss最大；
  > 其次梯度下降（**梯度用于更新参数**），找到最佳模型参数（所有层的模型参数，这一步和正常模型更新、梯度下降无异），使loss最小。
  >
  > <img src="https://img-blog.csdnimg.cn/20210425155836781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjM3ODUwOA==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 67%;" />
  >
  > attack就是将已算出的扰动加到embedding上的操作，目的就是看看怎么attack才能得到最佳扰动，对word-embedding层attack后，计算“被attack后的loss”，即对抗loss（adv_loss），然后据此做梯度上升，对attack的扰动r进行梯度更新。
  >
  > 

  > - **FGSM** (Fast Gradient Sign Method)
  >
  >   假设对于输入的梯度为：$g=\nabla_{x} L(\theta, x, y)$
  >
  >   那扰动肯定是沿着梯度的方向往损失函数的极大值走：$r_{a d v}=\epsilon \cdot \operatorname{sign}(g)$
  >
  > 
  >
  > - **FGM** (Fast Gradient Method)
  >
  >   FSGM是每个方向上都走相同的一步，FGM则是根据具体的梯度进行scale
  >
  >   $r_{a d v} =\epsilon \cdot g /\|g\|_{2}$
  >
  >   $g =\nabla_{x} L(\theta, x, y)$
  >
  >   步骤：
  >
  >   1. 照常计算前向loss，然后反向传播计算grad（注意这里不要更新梯度，即没有optimizer.step()）
  >
  >   2. 拿到embedding层的梯度，计算其norm，然后根据公式计算出r_adv，再将r_adv累加到原始embedding的样本上，即 x+r
  >
  >   3. 得到对抗样本； 根据新对抗样本 x+r, 计算新loss，在backward()得到对抗样本的梯度。由于是在step(1)之后又做了一次反向传播，所以该对抗样本的梯度是累加在原始样本的梯度上的
  >
  >   4. 将被修改的embedding恢复到原始状态（没加上r_adv 的时候）
  >
  >   5. 使用step(3)的梯度（原始梯度+对抗梯度），对模型参数进行更新
  >
  > 
  >
  > - ==**PGD**== (Projected Gradient Descent) （本次采用的）
  >
  >   FGM直接通过epsilon参数一下子算出了对抗扰动，这样得到的可能不是最优的。因此PGD进行了改进，多迭代几次，慢慢找到最优的扰动。FGM简单粗暴的“一步到位”，可能走不到约束内的最优点。PGD则是“小步多走”，如果走出了扰动半径为epsilon的空间，就映射回“球面”上，以保证扰动不要过大。
  >
  >   $x_{t+1}=\prod_{x+S}\left(x_{t}+\alpha g\left(x_{t}\right) /\left\|g\left(x_{t}\right)\right\|_{2}\right)$     >>> 1
  >
  >   $g\left(x_{t}\right)=\nabla_{x} L\left(\theta, x_{t}, y\right)$                                  >>> 2
  >
  >   在一步更新网络内（公式2），在S范围内进行了多步小的对抗训练（公式1），在这多步小的对抗训练中，对Word Embedding空间扰动是累加的。每次都是在上一次加扰动的基础上再加扰动，然后取最后一次的梯度来更新网络参数。
  >
  >   步骤：
  >
  >   1. 计算在正常embedding下的loss和grad（即先后进行forward、backward），在此时，将模型所有grad进行备份
  >
  >   2. 对抗攻击（K步的for循环）：反向传播（计算grad）是为了计算当前embedding权重下的扰动r，同时为了不干扰后序扰动r的计算，还要将每次算出的grad清零。
  >
  >      a.  如果是首步，先保存一下未经attack的grad。然后按照PGD公式以及当前embedding层的grad计算扰动，然后将扰动累加到embedding权重上；
  >
  >      b.  非第K-1步时：模型当前梯度清零；
  >
  >      c.  到了第K-1步时：恢复到step-1时备份的梯度（因为梯度在数次backward中已被修改）；
  >
  >   3. 使用目前的模型参数（包括被attack后的embedding权重）以及batch_input，做前后向传播，得到adv_loss、更新grad
  >
  >   4. 恢复embedding层 2.a 时保存的embedding的权重（注意恢复的是权重，而非grad；权重不变，只是下降方向有变化）
  >
  >   5. optimizer.step()，梯度下降更新模型参数。这里使用的就是累加了K次扰动后计算所得的grad。
  >
  >   应注意的是，在K步for循环的最后一步，恢复的是梯度，因为我们要在原始梯度上进行梯度更新，更新的幅度即”累加了K次扰动的embedding权重所对应的梯度“；而在attack循环完毕、要梯度下降更新权重前，恢复的则是embedding层的权重，因为我们肯定是要在模型原始权重上做梯度下降的。

  

  [对抗学习总结：FGSM-＞FGM-＞PGD-＞FreeAT， YOPO -＞FreeLb-＞SMART-＞LookAhead-＞VAT](https://blog.csdn.net/weixin_36378508/article/details/116131036)

## 经验总结

_**数据处理和数据增强真的很恨很恨很重要，折腾半天模型最多提升个0.01，数据上的改进可以轻轻松松提升0.02**_

1. 最后模型融合时，不要直接用求均值的办法，因为DeBERTa相对其他模型有压倒性优势，因此需要调高DeBERTa的比重
2. 有些大神在数据处理阶段，按照anchor和context进行分组，分组之后在同一组内的target全合并为target**s**（排除当前input自身的target），并拼接到每一个input之后（第一种输入形式+target**s**）
3. 预训练模型和自定义的下游网络结构可以使用不同的learning rate
4. CV也可以针对anchor而不是score进行分层采样；对于同一个网络架构，可以得到k个训练后的模型，根据OOF得分进行权重的选择

**其他需要进一步了解的操作**

1. 知识蒸馏
2. **?** Exponential Moving Average



