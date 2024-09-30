# more softmax

公式如下：

涉及到两个softmax操作

- safe softmax 用于解决溢出问题，但是会导致多一次访存。

    safe softmax的原理很简单，就是每个元素进行softmax之前首先减去最大值

- online softmax 用计算时间换取访存时间 

    online softmax相对来说复杂一点，详细的参照该论文。https://arxiv.org/pdf/1805.02867