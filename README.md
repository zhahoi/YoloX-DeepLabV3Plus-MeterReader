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

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git))
  
- [deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch.git))
  

训练好模型之后，使用仓库自带的转换脚本，可以将生成的权重转换成onnx格式。

随后，需要将onnx转换成ncnn，DeepLabV3Plus的转换比较顺利，但是YoloX转换遇到了一些小问题，感谢下面博客对解决我问题的帮助：

[使用ncnn部署yolox详细记录](https://www.bilibili.com/read/cv22065350/)

以上的操作，训练的模型都已经转换成ncnn格式了，接下来就是要编写c++代码。

##

## 转

因为我的C++代码水平很差，所以我都是边抄边改边调试，最后完成大概。这里十分感谢以下的几个仓库的贡献，代码中大部分的代码都是抄这里面的：

- [Yolov5-DeepLabV3Plus-MeterReader](https://github.com/xinglunancv/Yolov5-DeepLabV3Plus-MeterReader) (抄了指针读数的代码)
  
- [MeterReader](https://github.com/zhuyushi/MeterReader.git)) (抄了部分后处理的代码)
  
- [Person_Segmentation](https://github.com/runrunrun1994/Person_Segmentation.git)) (抄了DeepLabV3Plus部分的ncnn处理代码)
  
- [ncnn](https://github.com/Tencent/ncnn/blob/master/examples/yolox.cpp)) (抄了YoloX的ncnn处理代码)
  

对输入图像进行不失真的resize，感谢以下博客：

[C++ - Yolo的letterbox图片预处理方法，缩放图片不失真](https://www.stubbornhuang.com/2728/))

## 合

经过以上三步操作，代码基本上是可以在Visual Studio运行了，但是遇到一个问题是因为载入了两个深度学习模型，运行起来比较卡顿，这对我来说是比较头疼的问题，因为语义分割我暂时找不到更合适的模型，即使是有轻量话模型，也没法丝滑转换成ncnn，可能ncnn后处理代码也没有。因为自己水平有限，暂时不考虑。

以上的代码经过自己的测试可以执行成功，但是仪表的识别效果感觉还是一般，我总觉得是语义分割数据集太少了，后续应该需要再扩充，检测效果才会好。

上传这个仓库仅大家参考学习，提供一点灵感，后续有时间会继续尝试对该代码进行完善。
