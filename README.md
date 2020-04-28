### bayes_spam
**一个基于贝叶斯用于预测一个中文邮件是否为垃圾邮件的预测模型，同时还提供了用于预测提供服务的
基于thrift rpc的服务端程序**

关键字：贝叶斯分类 垃圾邮件分类 rpc

keywords: bayes classifier, spam recognize, rpc

#### 使用说明
1. 环境要求
    1. python 3.x
    2. 对于本项目所需要的第三方库，请直接使用如下命令完成安装：
        * e.g. `bash pip_libs.sh -M` 
        * 注：由于只提供shell脚本，没有提供bat/cmd脚本，故对于windows用户，你可以使用
            git自带的git bash来运行这个脚本。
    3. 对于简单的模型训练与使用，请参考examples.py
2. 使用rpc功能完成客户端发送文件，服务器端完成预测并返回预测结果
    * 实现了python server，java cli，完成从java端传输文件，python server
        进行预测并返回结果
    * 启动服务器：
        1. 保证执行了 环境要求中的第三方库安装 
        2. 首先保证本地已有预测模型（首次通过运行examples.py中的train函数完成模型构建并将模型存储到文件中）
        3. 运行rpc/server.py的main方法即可启动服务器
        
#### 模型调参
1. d_type
    * 用于设置计算过程中数据的精度
    * 默认np.float32
2. p_class_0与p_class_1
    * 分别代表p(任意一个文件是正常邮件)，p(任意一个文件是垃圾邮件)
    * 默认值: 0.5, 0.5
3. 设置在预测过程中的对邮件预测影响最大的k个单词，从而计算就不需要计算所有单词概率对预测的影响
    * 也就是model.set_threshold(k), k=-1时表示取所有单词
    * 默认值-1
    * 实验中时取-1值，在测试集合预测效果最好，若你要提升预测速度可以修改这个值为较小值（不保证准确度）
    * 自我感觉这个参数是个鸡肋，甚至计算还更慢
* 全部在默认值的情况下，在测试集合上的准确度高于98%

#### 配置文件
见项目下的conf/conf.ini，各项配置的解释见注释
        
#### 注意

先启动python的服务器，再启动java的客户端
java客户端启动运行见以下README

https://github.com/ZhangQi1996/bayes_spam_rpc

OR

https://gitee.com/ChiZhung/bayes_spam_rpc
        
    
