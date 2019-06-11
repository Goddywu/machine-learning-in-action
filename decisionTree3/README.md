## 简介

数据有相当多的维度，构造决策树最重要就是选择作为枝节点的维度。
<br>我们优先选择信息增益最大的维度(也可以使用基尼不纯度Gini impurity)，即ID3算法。

#### 决策树-ID3算法
<br>infoGain = baseEntropy - newEntropy
<br>香农熵(shannon entropy)为：
<br>![](/images/shannon_entropy.jpg)

数据量较少、特征值很多时，很容易造成**过度匹配**(over fitting)的问题，
第9章会有裁剪决策树的算法。