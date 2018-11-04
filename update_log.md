

##### 2018.10.11
- 增加 config/common.py 文件，用于罗列当前可用模型 并且可以解耦合。
- 增加 Transformer 模型。（单纯增加，本人还未完全理解该模型）

##### 2018.10.10
- 重构 fasttext 模型代码。同时增加facebook版本和纯python版本的支持，增强平台通用性。
- 增加 fasttext_package文件夹，用户可根据自己的环境安装相应版本的 fasttext。
- 重构 model_helper/run_one_epoch 方法，使之更为通用。

##### 2018.10.09
- 大量重构 classifier_models/classifier.py 的代码，打包了一些layer操作。以后添加新模型更加方便。
- 增加 HAN 模型。
- 增加 tests 文件夹，用于存放测试脚本。

##### 2018.10.08
- 增加 fasttext 模型。
- 增加 experiment 文件夹，用于存放实验结果。

##### 2018.09.27  
- 增加命令行参数运行方式。  
- 优化 README.md 中的句子通顺度，并为 README.md 创建目录。  

##### 2018.09.26  
- 将3种执行mode从entrance.py中分离出来，提升可阅读性。  
- 修改了bug ; 更新了README.md  
- 增加了自带数据文件：粤语句子数据文件。  

##### 2018.09.21  
- 补上了许多注释和文档，重构了项目结构。  
- 添加 README.MD 和 update_log ；将3种执行模式单独写成方法  
- 添加了一些运行程序时的提示语句  
- 增加 IndRNN 模型。  

##### 2018.09.18  
- 初步搭建好了项目结构。实现了 AttnBiLSTM 分类模型。  
