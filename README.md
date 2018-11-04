# 文本分类模型整理
- [简介](#简介)
- [Requirement](#Requirement)
- [使用步骤](#使用步骤)
    - [1. 数据集准备](#数据集准备)
    - [2. 选择模型](#选择模型)
    - [3. 运行entrance.py](#运行entrance.py)
    - [4. 开启tensorboard可视化](#开启tensorboard)
- [三种执行模式介绍](#三种执行模式介绍)
- [目前提供的模型](#目前提供的模型)
- [项目文件结构](#项目文件结构)

## 简介
实现一些state-of-the-art文本分类模型，并提供方便的调用方式。  

## Requirement
- Python3
- tensorflow-gpu >=1.4

## 使用步骤
### <span id="数据集准备">1. 数据集准备 
- 数据集存放位置：  
    数据文件放在input_data文件夹中。放置样例参见input_data/sample_data文件夹。   
        input_data文件夹中还自带了PTB数据集dbpedia_data（已转换为fasttext输入数据形式。没有保存原始数据）；  
        还有粤语句子数据集cantonese_data
    
- 数据样本格式：  
    数据需要预处理为 形如fasttext输入数据形式。例子：
    ```
    __label__1 Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. 
    __label__1 A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.
    ```
- 数据集命名格式：  
训练集文件命名为data_train.txt   
测试集文件命名为data_test.txt  
验证集文件命名为data_valid.txt（如果没有验证集文件，将会从测试集中按照9:1分割出验证集）  
待预测文件命名为data_predict.txt（只在pred模式才需要该文件。与其他文件的区别是没有__label__项）  

### <span id="选择模型">2. 选择模型
从目前提供的模型中选择想要使用的模型，并修改对应模型的配置文件。
模型配置文件对照表如下：

模型 | 配置文件
---|---
1. Attention-Based Bidirection LSTM | config/attn_bi_lstm.py
2.  Independently RNN | config/ind_rnn.py
3. Hierarchical Attention Networks | config/hierarchical_attn_net.py
6. fasttext | config/fasttext.py

*注意：具体需要修改的是配置文件中的(model_name)_config类，因为实际载入的是(model_name)_config类*  

### <span id="运行entrance.py">3. 运行entrance.py
三个运行参数如下：
- input_path ：设置数据集路径。数据集都应该放在input文件夹下。
- mode ：设置执行模式。三种执行模式可选：train, eval, pred
- model_idx ：分类模型的序号

运行样例：（选择粤语数据集，选择第一个模型，执行训练模式）
```
python entrance.py --input_path=input_data/cantonese_data --mode=train --model_idx=1
```

### <span id="开启tensorboard">4. 开启tensorboard可视化（可选）
训练过程中，可以开启tensorboard观察训练情况。以AttnBiLSTM模型为例，查看训练情况的方法： 

    tensorboard --logdir=(项目所在目录)/text_classification/classifier_log/attn_bi_lstm/train
    
    
## 三种执行模式介绍
- train 模式  
    用于训练分类模型。最少需要提供训练数据data_train.txt和测试数据data_test.txt。  
    将会在output/vocab_file文件夹中生成字典文件vocab.txt。  
    将会在classifier_log文件夹中保存每个模型的训练进度和summary  
    如果训练完模型后更改了配置文件中的参数，一般需要先删除classifier_log下对应的模型名文件，
    
- eval 模式  
    用于评估分类模型的性能。需要提供测试集数据data_test.txt。  
    
- pred 模式  
    用于预测新的数据文件的类别。需要提供待预测数据data_predict.txt。   
    
- **eval 模式和 pred 模式需要注意的点**  
    **必须先运行train模式，否则会报错。**   
    不能改变训练模型时对应的配置文件的参数，否则会出现问题。  
    将会读取train模式保存的字典文件。  
    将会读取train模式训练的模型进度，用于预测。

## 目前提供的模型
#### 1. Attention-Based Bidirection LSTM
Paper: [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)  
模型实现详见 classifier_models/attn_bi_lstm.py

#### 2. IndRNN
Paper: [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)  
模型实现详见 classifier_models/ind_rnn.py

#### 3. Hierarchical Attention Networks
Paper: [Hierarchical Attention Networks for Document Classification](http://aclweb.org/anthology/N16-1174)  
模型实现详见 classifier_models/hierarchical_attn_net.py

#### 4. Transformer
Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
模型实现详见 classifier_models/transformer.py

#### 5. Adversarial Training Methods For Supervised Text Classification
Paper: [Adversarial Training Methods For Semi-Supervised Text Classification](http://arxiv.org/abs/1605.07725)  
模型实现详见 classifier_models/adversarial_abblstm.py （未完成）

#### 6. fasttext
Paper: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
（需要先参照fasttext_package中的README.md，安装好fasttext）  
模型调用详见 classifier_models/fasttext.py

## 项目文件结构
- config文件夹：存放每个模型对应的配置文件，每个配置文件里有两个配置类：
    - (model_name)_config类 是程序实际载入的配置。
    - (model_name)_common_config类 是供参考的默认配置。

- classifier_log文件夹：存放每个模型的训练进度和summary. 训练完模型后才会出现该文件夹。

- classifier_models文件夹：存放每个模型基于tensorflow的实现。

- experiment文件夹：存放模型实验结果。

- fasttext_package文件夹：存放两个版本的fasttext安装包。

- input_data文件夹：存放训练数据。

- output文件夹：存放输出部分。包括字典和预测值。  
    - 当以train模式运行时，会在output/vocab_file文件夹下生成字典文件vocab.txt。  
    **注意：字典文件会保存训练集中所遇到的所有单词。不过实际运行过程中，会根据配置文件里的max_vocab_size参数，限制程序能“看到”的字典大小。**
    - 当以pred模式运行时，会在output文件夹下生成预测值文件。
    
- tests文件夹：存放一些测试脚本。

- utils文件夹：存放常用的方法和类。包括data_helper, model_helper, vocabulary类 等。

- entrance.py文件：项目的入口。

## 项目思路介绍：（想读源代码的先可以看看）
因为。。。