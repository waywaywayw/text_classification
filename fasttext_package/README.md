# fasttext包的安装方法
鉴于fasttext安装包的复杂情况。还是单独说明一下fasttext包安装方法吧。


## for window平台
首先，直接 pip install fasttext 安装的fasttext包目前（2018.10.10）是不能用的。。  
然后，因为pybind11包安装不方便，所以facebook原版在window平台也不好安装。。
- 只能在github上下载fasttext的纯python版本：
https://github.com/salestock/fastText.py 
 
- 已下载好的文件在：  
fasttext_package/fastText.py-master.zip  

将项目文件解压然后pip install . 安装即可。  
显示版本号：0.8.3。

## for linux平台
- 在github上下载fasttext的facebook原版：  
https://github.com/facebookresearch/fastText 
- 已下载好的文件在：  
fasttext_package/fastText-master.zip

将项目文件解压，然后pip install . 安装python包；最后 make 安装fasttext。  
显示版本号：0.8.22。
