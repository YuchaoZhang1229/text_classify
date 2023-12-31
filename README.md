### 一、模型结构创建
    1.1. 输入输出理解
        输入：
            [N,T]: 评论原始文本 --> 分词 --> 词典映射/id序号转换 --> 形成一个数值类型的文本向量 --> 组合批次数据输入模型
                N表示N个文本，T表示每个文本由T个单词id组成。
        输出：
            [N,2]: N表示N个文本，2表示每个文本属于两个类别的置信度
    1.2. 模型结构
        a. 基于全连接的网络
### 二、搭建整个的训练体系
    1.1 整个代码的结构
        a. 数据加载
            NLP: 分词、序号id转换、dataloader的构建
        b. 模型、优化器、损失的创建
        c. 迭代训练、评估
        d. 模型持久化
### 三、模型部署
    基于Python Flask Web的模型部署
        -1. 将模型转换为静态结构：Torch Script、ONNX
            https://pytorch.org/docs/stable/jit.html#torchscript
            https://pytorch.org/docs/stable/onnx_torchscript.html#module-torch.onnx
        -2. 本地编写模型的预测代码
            仅允许存在最少的对外调用接口/方法，一般情况下两个：初始化方法、predict预测方法
        -3. 编写Flask Web外壳
        -4. 部署测试
            -4.1 将代码以及模型文件copy到linux服务器上，并在linux安装好运行环境，最终通过python命令启动服务即可；
            -4.2 通过python requests库远程访问linux上的服务进行check
