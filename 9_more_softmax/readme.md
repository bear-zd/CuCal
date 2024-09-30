# more softmax

公式如下：

涉及到两个softmax操作

- safe softmax 用于解决溢出问题，但是会导致多一次访存。

    safe softmax的原理很简单，就是每个元素进行softmax之前首先减去最大值

- online softmax 用计算时间换取访存时间 

    online softmax相对来说复杂一点，详细的参照该论文。https://arxiv.org/pdf/1805.02867。

    整体来说公式就是两步：

    第一步同时结合了safe softmax的最大值和求和：

    $$m_j = max(m_{j-1}, x_j)$$

    $$d_j = d_{j-1}\times e^{m_{j-1}-m_j} + e^{x_j - m_j}$$

    第二步就是逐元素进行操作

    则最终的元素值就会变成:

    $$y_j = \frac{e^{x_i-m_V}}{d_V}$$

    从而把访存数量减少到了和原来一样的水平

    相较于源码，使用了reduce相关操作进行。

