# softmax

公式如下：

$$softmax(\bold x) = \frac{e^x_i}{\sum^K_{j=1} e^x_j} \space for\space i=1,2,...,K$$

那么相对于前面的规约求和和点积其实相差不大，首先是一个逐点进行指数计算的操作，随后是一个规约求和的操作。