> Liu, Wei; Anguelov, Dragomir; Erhan, Dumitru; Szegedy, Christian; Reed, Scott; Fu, Cheng-Yang; Berg,Alexander C.SSD: Single Shot MultiBox Detector.eprint arXiv:1512.02325.12/2015
# 一、笔记
![image](http://m.qpic.cn/psb?/V12DPMma00AtxB/0ukGHA8nEqrCBCSg.SxWW2qp9sTufJfkKuYOoou1Xek!/b/dDMBAAAAAAAA&bo=9AMkAfQDJAEDByI!&rf=viewer_4)

SSD的base network基于VGG-16（分类层之前的所以部分）。其创新之处在于增加了如下结构：
1. 多维度特征图（Multi-scale feature maps for detection）
> 加入了一些卷积层，目的是在不同的维度/范围进行预测检测并减小大小。（*？？？表示不理解*）
2. 卷积预测器（Convolutional predictors for detection）
3. 默认框和长宽比（Default boxes and aspect ratios）
# 二、参考阅读
[SSD: Single Shot MultiBox Detector 深度学习笔记之SSD物体检测模型](https://www.sohu.com/a/168738025_717210)

[RCNN学习笔记(10)：SSD:Single Shot MultiBox Detector](https://blog.csdn.net/u011534057/article/details/52733686)
