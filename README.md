## 多模态情感分析

* 给定配对的文本和图像，预测对应的情感标签



## 准备工作

本项目中代码基于Python3实现，以下是代码需要用到的库：

* numpy==1.23.5
* Pillow==10.0.0
* torch==2.0.1+cu118
* transformers==4.24.0



你可以通过运行以下命令安装需要的库：

~~~python
pip install -r requirements.txt
~~~





## 代码结构

~~~
│  config.py------------------------配置文件
│  img_only.py----------------------仅图片模型
│  main.py--------------------------调试用
│  model_multi----------------------模型文件
│  multi.py-------------------------多模态混合模型
│  multiData.py---------------------得到多模态混合数据集
│  prediction.txt-------------------预测文件
│  runUtils.py----------------------训练工具
│  txt_only.py----------------------仅文本模型
│
├─data_pre
│  │  test_without_label.txt--------测试数据
│  │  train.txt---------------------训练数据
│  │
│  ├─data
│  └─input
│          getData.py---------------读取与预处理数据文件
│          testData.json------------预处理后的测试数据
│          trainData.json-----------预处理后的训练数据
│
├─img_pro
│      image.py---------------------得到图像数据集
│      img_config.py----------------配置文件
│      ViT.py-----------------------ViT
│
└─txt_pro
        bert.py---------------------BERT
        text.py---------------------得到文本数据集
        txt_config.py---------------配置文件
~~~



## 运行方式

### 使用方法与GitHub地址

* 若要运行混合模型，则运行 `multi.py` 

* 若要运行单文本数据的模型，则运行 `txt_only.py` 

* 若要运行单图像数据的模型，则运行 `img_only.py` 

* 需要注意的是，由于运行速度较慢，以上代码的训练部分都已经注释掉了，如果需要重新训练，将main改为以下内容即可

* ```python
  if __name__ == '__main__':
      run()
      test_mul()
      predict_mul()
  ```



## 参考资料

* [万字综述！从21篇最新论文看多模态预训练模型研究进展_kaiyuan_sjtu的博客-CSDN博客](https://blog.csdn.net/Kaiyuan_sjtu/article/details/121391851)
* [多模态情感分析简述 | 机器之心 (jiqizhixin.com)](https://www.jiqizhixin.com/articles/2019-12-16-7)

