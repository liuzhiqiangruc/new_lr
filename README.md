主要提供一个轻量级的deep learning实现。
Loss function 与LR一致，求二项分布的极大似然
因此顺带着提供了一个LR的实现。

代码下载之后直接执行make 命令，会在当前目录下生成两个可执行文件：
1. lr
2. deeplr

其中 deeplr 为一个全连接的单隐层分类器。直接执行命令，会有参数说明。

训练数据格式说明：
1. 对于onehot的离散特征：   
   label\tfeature1\tfeature2\tfeature3....       
   在命令中使用 -b 1 参数

2. 对于有值得连续值特征：    
   label\tfeature1\tvalue1\tfeature2\tvalue2...      
   在命令中使用 -b 0 参数



