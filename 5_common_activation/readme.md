# common activation

前面使用诸多点对点的核函数来熟练操作，但是上面的代码都是不工整甚至有些混乱，有很多类型检测等都没有进行，因此这里使用宏来实现一些激活函数的实现并很好的打包。编写完成宏之后就可以尽可能多的进行扩展了，同样的，也可以学习一下CUDA learning里面对于数据类型和数据并行的扩展

相关的激活函数：https://en.wikipedia.org/wiki/Sigmoid_function