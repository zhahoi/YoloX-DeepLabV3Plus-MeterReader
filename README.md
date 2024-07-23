# YoloX-DeepLabV3Plus-MeterReader

[](https://github.com/zhahoi/YoloX-DeepLabV3Plus-MeterReader#yolox-deeplabv3plus-meterreader)

使用YoloX+DeepLabV3Plus实现仪表的检测、指针表盘分割和刻度读数识别（借助ncnn框架）

## 起

创建这个仓库的原因是因为公司需要做一个指针仪表识别的项目，因为没有其他人做，然后自己就被派遣了这个任务。因为之前没做过类似的，所以就在网上各种搜索别人做过的，有用传统算法做的，当然也有用深度学习做的。检索一番之后发现，还是先用目标检测再用语义分割这种做法比较靠谱。

之前搜到百度的paddle paddle有用目标检测+语义分割做好的案例，但是基于paddle比较难用，直接用别人的方案自己的提升没那么大，于是就又继续搜索。然后我在GitHub找到这种方案的类似几个实现，想着改改别人的代码应该也有戏，于是便投入了该项目。

## 承

仪表读数算法包含目标检测（先检测出数字仪表的位置，进行裁剪）和语义分割（对裁剪出的图片进行语义分割，分割出表盘指针和刻度），随后使用传统算法进行处理，然后读数。具体的原理可以参考[工业表计读数]([工业表计读数 &mdash; PaddleX 文档](https://paddlex.readthedocs.io/zh-cn/release-1.3/examples/meter_reader.html))。

目标检测和语义分割的数据集，大部分取自飞桨提供的数据，我自己也制作了一些我需要识别的仪表的数据，我将其转换成了VOC格式，可以通过以下百度云链接下载：链接：https://pan.baidu.com/s/16oTwL0OpL0AV07tJeAB-8g?pwd=1234 
提取码：1234

模型训练的话，我是直接使用了以下两个仓库进行了训练：

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)
  
- [deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch.git)
  

训练好模型之后，使用仓库自带的转换脚本，可以将生成的权重转换成onnx格式。

随后，需要将onnx转换成ncnn，DeepLabV3Plus的转换比较顺利，但是YoloX转换遇到了一些小问题，感谢下面博客对解决我问题的帮助：

[使用ncnn部署yolox详细记录](https://www.bilibili.com/read/cv22065350/)

以上的操作，训练的模型都已经转换成ncnn格式了，接下来就是要编写c++代码。

##

## 转

因为我的C++代码水平很差，所以我都是边抄边改边调试，最后完成大概。这里十分感谢以下的几个仓库的贡献，代码中大部分的代码都是抄这里面的：

- [Yolov5-DeepLabV3Plus-MeterReader](https://github.com/xinglunancv/Yolov5-DeepLabV3Plus-MeterReader) (抄了指针读数的代码)
  
- [MeterReader](https://github.com/zhuyushi/MeterReader.git) (抄了部分后处理的代码)
  
- [Person_Segmentation](https://github.com/runrunrun1994/Person_Segmentation.git) (抄了DeepLabV3Plus部分的ncnn处理代码)
  
- [ncnn](https://github.com/Tencent/ncnn/blob/master/examples/yolox.cpp) (抄了YoloX的ncnn处理代码)
  

对输入图像进行不失真的resize，感谢以下博客：

[C++ - Yolo的letterbox图片预处理方法，缩放图片不失真](https://www.stubbornhuang.com/2728/)

## 合

经过以上三步操作，代码基本上是可以在Visual Studio运行了，但是遇到一个问题是因为载入了两个深度学习模型，运行起来比较卡顿，这对我来说是比较头疼的问题，因为语义分割我暂时找不到更合适的模型，即使是有轻量话模型，也没法丝滑转换成ncnn，可能ncnn后处理代码也没有。因为自己水平有限，暂时不考虑。

以上的代码经过自己的测试可以执行成功，但是仪表的识别效果感觉还是一般，我总觉得是语义分割数据集太少了，后续应该需要再扩充，检测效果才会好。

上传这个仓库仅大家参考学习，提供一点灵感，后续有时间会继续尝试对该代码进行完善。


## 补充
[2024.06.05]更新：增加了另一种检测仪表的方案，算法原理参照的是[MeterReader](https://github.com/zhuyushi/MeterReader.git)这个仓库，该方案可以根据仪表的刻度和量程自动计算结果，每次无需指定仪表类型。但是第一种仪表检测方案每次只能检测一种，不太方便。

检测结果示意图
![Dingtalk 20240605170728](https://img.picgo.net/2024/06/05/Dingtalk_202406051707282f620109b6ce2484.jpg)
![Dingtalk_20240606085046.jpg](https://vip.helloimg.com/i/2024/06/06/666107ebedd82.jpg)

[2024.06.29]更新：最近在部署代码的时候发现很多的问题，尤其是DeepLabv3-Plus输出的结果时常十分诡异地出错。查了很久，才发现原因可能是参考仓库的预处理与后处理与pytorch训练模型的预处理与后处理不匹配导致的。因此，现在尝试更新c++代码的预处理与后处理，结果还没验证是否正确，先进行了修改。果然，不动脑子光copy别人的代码，还是可能存在问题的。希望看到这个仓库的朋友，也可以根据自己的实际需要，调整代码的内容。我只是一个刚用c++没多久的菜鸡，本仓库的代码估计有很多很多不规范的地方，以后会慢慢想办法修改，让其更标准科学。希望大家参考的时候多注意，避免被我带进坑里面。

[2024.07.02]更新：之前提到DeepLabV3-Plus输出的结果诡异地出错，其实一直也没解决，昨天去ncnn官方库下面搜issues发现也有其他人和我遇到同样的问题，因此可以断定应该不是我预处理与后处理的问题。因为DeepLabV3-Plus输出结果不稳定，所以我重新选择了另外的Unet模型用来分割指针仪表的表盘方便读数。我重新创建另一个仓库上传了使用Unet模型来分割指针仪表表盘的ncnn推理代码，同时验证了一下没有任何推理错误。如果你自己推理的时候发现了和我同样的问题，可以参考我新建的另一个仓库，希望可以帮助你解决语义分割结果偶发错误的问题。
- [Unet_ncnn](https://github.com/zhahoi/Unet_ncnn.git)

[2024.07.23]更新：为了满足实时性的要求，这一套算法重新使用NanodetPlus和MobilenetV3-LRASPP重新实现了一下，可以实现实时检测和读数。重新创建了一个代码仓库供大家参考。
- [NanodetPlus-MobilenetV3-LRASPP-MeterReader](https://github.com/zhahoi/NanodetPlus-MobilenetV3-LRASPP-MeterReader)
