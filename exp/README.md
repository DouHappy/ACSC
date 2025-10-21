# 实验性功能文件夹
这个文件夹里面存放的是探索性的实验代码，因为需要频繁迭代，所以不放到源文件中

# 目前主要的内容有：

## statistics: 获取大模型推理时的attention分布。
输入: 
a. 模型地址
b. 分析输入地址（CSC数据集）
输出：
attention_batch_{start_id}_{end_id}.npz
{
    "start_id": int,
    "end_id": int,
    "attentions": np.array([num_head, batch, num_layer, seq_len, seq_len])
}

##  analyze.py：分析attention分布的特征
需要配置分析文件来获取特定的attention的统计结果。

## analyzer_torch.py：出于分析性能考虑对analyze.py进行的重构实验
主要瓶颈不在于numpy或者torch，后期没有继续维护。

## analyzer.py：抽象化了attention分析的流程。
每个位置获取函数对应一个exp，即我们想要分析那些位置的实验。
每个位置的attention数据有多个数据处理方式，例如mean，max等

根据步骤主要有三个函数：

### pre_extract() 

预先把需要分析的token2tokens的attention保存下来。
需要自定义一个函数，对于一个csc输入和输出返回需要分析的attention有哪些。
保存结果为：
exp_name.npy: [num_qid, num_head, num_layer, number_kids]
第一维度代表一个q，最后一个维度代表一个k

### analyze()

在pre_extract的基础上需要配置分析函数，即对于每个qid的kids个操作要做什么样的数据处理。
将上述的数据处理结果（一个float64）取平均值，作为最终结果。
对于每个

###  statistic_distance()

使用不同的距离函数计算正例和负例到我们提取的特征之间的距离。
分析这样的距离函数和特征能否把正例和负例很好的区分开，P-R-F1曲线。