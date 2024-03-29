# [PRCV 2018](https://prcv-conf.org/2018/comp_list_05?from=singlemessage&isappinstalled=0)

> 更新 20180815 V6
# 一、主要更新
1. 将分类损失的权重调大；
2. 进一步训练。
# 二、评估结果
```SSD_Inception_V4评估结果——100k stpes：```
```
INFO:tensorflow:Losses/Loss/classification_loss: 54.618763
INFO:tensorflow:Losses/Loss/localization_loss: 0.905538

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/100: 0.002080

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/103: 0.000518

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/104: 0.009259

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/108: 0.003309

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/11: 0.002161

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/111: 0.065599
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/112: 0.006323

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/113: 0.266377
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/114: 0.072111

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/115: 0.001606
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/116: 0.047629

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/117: 0.021858
I
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/148: 0.009390

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/154: 0.015385
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/41: 0.033333

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/31: nan
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/105: nan

INFO:tensorflow:PascalBoxes_Precision/mAP@0.5IOU: 0.003053

```
# 三、结果分析

1. mAP居然下降了，这个还是另我挺意外的；
2. 能准确识别的类由15类上升到17类；
3. 回归损失下降了将近20%，说明定位精度有所提高。

# 四、问题分析及解决思路

我还是认为mAP低于预期是由于分类精度过低导致的，现在有2个解决思路：
1. 向主办方请求用于战车/坦克分类的数据集，从头训练用于特征向量提取的inception V4网络；
2. 将所有战车/坦克的标签改为类1，将检测转变为单纯的战车/坦克检测问题。

> 更新 20180812 V5

# 一、主要更新内容
1. 将提取特征向量的cnn网络换成了更高级的inception V4；
2. 对模型进行了定量评估。

# 二、评估结果
```1.SSD_Inception_V3评估结果：```
```
INFO:tensorflow:Losses/Loss/classification_loss: 13.053458
INFO:tensorflow:Losses/Loss/localization_loss: 1.474766

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/1: 0.001048
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/100: 0.038720
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/101: 0.011621
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/102: 0.002060
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/103: 0.016934
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/104: 0.084230
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/106: 0.021555
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/107: 0.004495
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/108: 0.013519
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/109: 0.015791
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/11: 0.004934
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/118: 0.006469

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/31: nan
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/105: nan

INFO:tensorflow:PascalBoxes_Precision/mAP@0.5IOU: 0.001171
```

```2.SSD_Inception_V4评估结果：```

```
INFO:tensorflow:Losses/Loss/classification_loss: 12.716391
INFO:tensorflow:Losses/Loss/localization_loss: 1.102177

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/1: 0.074396
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/10: 0.021739
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/100: 0.265250
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/101: 0.156168
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/102: 0.006062
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/103: 0.009645
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/104: 0.019608
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/107: 0.003719
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/108: 0.004348
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/11: 0.005988
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/111: 0.009299
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/112: 0.012327
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/113: 0.000601

INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/31: nan
INFO:tensorflow:PascalBoxes_PerformanceByCategory/AP@0.5IOU/105: nan

INFO:tensorflow:PascalBoxes_Precision/mAP@0.5IOU: 0.003117
```
```3.相关说明：```
为使评估结果具有说服力，两个模型均训练20000step，且Batch_size=2。
# 三、结果分析
从评估结果可以看出，```SSD_Inception_V4```的表现明显优于```SSD_Inception_V3```。其能够识别出15类坦克（```SSD_Inception_V3```为14类），mAP差不多是```SSD_Inception_V3```的3倍。

# 四、存在的问题及解决思路
```1.问题：```虽然```SSD_Inception_V4```的mAP差不多是```SSD_Inception_V3```的3倍，但还是非常的低，只有0.003117。

```2.分析：```mAP过低，主要原因是分类不准确，实际上模型对图片中坦克位置检测的效果还是不错的。

```3.解决思路：```
1. 进一步训练；
2. 提高分类loss的权重。

---


> 更新 20180721 V4
# 一、主要更新内容
1. 将提取特征向量的cnn网络换成了更高级的inception V3；
2. 改善分类错误的问题。

# 二、检测效果
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/EzTwWqlLoYxNh.*SXXH7z8ThZgm3DnZOMR22Lgc.KMA!/b/dEMBAAAAAAAA&bo=mgLVAZoC1QEDR2I!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/DOkjc3aGg7P3HrBHMRI*.hYAi3t8d0EwOg5W6KARD88!/b/dDEBAAAAAAAA&bo=yQLEAckCxAEDNxI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/JYVHf2y9*i0..nULHZBfnos8IeVylgTuPHqtN1AjSe8!/b/dEEBAAAAAAAA&bo=TwLVAU8C1QEDJwI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/JkOpktnZgm4ndniJBsE4voSQUj70cUgxcwnDmsGvxss!/b/dC8BAAAAAAAA&bo=3wHVAd8B1QEDJwI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/yFfbk*oHS54ZXw29XifqxftZWTiHk2yRDX7LIyWKgqc!/b/dC8BAAAAAAAA&bo=yQKdAckCnQEDNxI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/11ZwyEU1vM19l*HLE76MqCwes26WuyXMIozd3IO8.gA!/b/dFYBAAAAAAAA&bo=uALVAbgC1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/36G9vBQdcH0jOr4CL2ItIBm*Chq8FGNJM1B6yIv19BA!/b/dDIBAAAAAAAA&bo=bwLVAW8C1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/zSEFWsw6urGeVr*Hr919tFv.diAs28JJKoHF24sh.Zc!/b/dDIBAAAAAAAA&bo=yQJnAckCZwEDJwI!&rf=viewer_4" width="40%" height="40%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/BH3xBxsbeJAdL2G9RTeBbboT.OKKb3OnlubfdtQ4Ejs!/b/dIMAAAAAAAAA&bo=bwLWAW8C1gEDNxI!&rf=viewer_4" width="40%" height="40%" />

# 三、存在的问题
1. 类别还是会判错；
2. 检测效果还是不理想。

# 四、问题分析及解决思路
1. 类别会判错这是分类的问题，采用更高级的特征向量提取网络应该能改善问题；
2. 利用坦克数据集而不是voc2007对base model进行训练，也可能改善模型性能。

# 五、下一阶段目标
1. 写一个script对模型的mAP进行评估；
2. 争取利用inception V4的SSD进行训练。

---


> 更新 20180720 V3

目前在Tensorflow detection model zoo中，基于SSD框架的目标检测神经网络中，COCO mAP[^1]最高的是ssd_resnet_50_fpn_coco，其COCO mAP[^1]为35。其用于提取特征向量的base模型，在ILSVRC-2012-CLS 上的Top-1 Accuracy=75.2。从理论上来说，利用在ILSVRC-2012-CLS 上的Top-1 Accuracy=78.0的Inception V3作为base模型的SSD，能取得更好的检测效果。

下一阶段的任务是，利用```DetectionModel```接口，打造ssd_Inception V3网络，用于训练。

---

> 更新 20180714 V2
# 一、主要更新内容
1. 对主办方给的数据进行筛查，剔除问题数据；
2. 对处理数据的scripts进行了修改，更正了神经网络无法识别图像类别的bug；
3. 进一步训练，验证模型效果。
# 二、检测效果
<figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/fQJb2rqjwuwwXb.Rq1.rKEaA1eUs*s.5WgzFlJGOyLU!/b/dDABAAAAAAAA&bo=yQJ.AckCfgEDByI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/4Z5Opdecu.gFxtPM5qySQmPP.ehsZEdzkEUD*gIwVfw!/b/dDMBAAAAAAAA&bo=bwLWAW8C1gEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure>
<figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/sraq3A9K9VURba92inDwjyX94XzPTf270uy5aT3IyTI!/b/dC0BAAAAAAAA&bo=OwLVATsC1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/N.SFwbOpYPas7523vtpIolRGorTo6orXUc*OUjnhJ1I!/b/dDIBAAAAAAAA&bo=yQKbAckCmwEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure>
<figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/cke6njQLKGW6xb2BnB84YLS1ZQHx5dY5xwYvGALw8ag!/b/dDIBAAAAAAAA&bo=vgLVAb4C1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/3OJkxnIjmyR6pvFyJhrd4bop7gzlEP85vGAEH6rhmPo!/b/dEABAAAAAAAA&bo=mgLVAZoC1QEDR2I!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/xV5rIK4hY65yZKfrFU8t5wPK2LzZly7K90Bc5FZrS2Q!/b/dDEBAAAAAAAA&bo=SALWAUgC1gEDR2I!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/ZEQ6N878sYU8x9Se7*1NsTXRSo58dbIyzcXvLB5ODLQ!/b/dC4BAAAAAAAA&bo=vALVAbwC1QEDJwI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/gRqvs.dwkJeVYuiYKob4vhxhpAX7kFDzSLFPuukboq8!/b/dDEBAAAAAAAA&bo=OgLWAToC1gEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/o9z2DZetyUTtj65JAyBSIq4mxF5y4VbGk*b*SitybzQ!/b/dDMBAAAAAAAA&bo=yQJ.AckCfgEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/l47PLHm8czFCpDHPBY9oAwgsUN50313uKsaEmRFvogA!/b/dFkAAAAAAAAA&bo=gALVAYAC1QEDFzI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/2xqDZXL2mpGS3wuMIQDtLtoh8ym*upa7.1v4hJmsxkk!/b/dFcAAAAAAAAA&bo=wwLWAcMC1gEDR2I!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/yc4OdVtdST.RV8jKEDfygT1rlrTxsnfsNv8fY40OwWk!/b/dDIBAAAAAAAA&bo=twLVAbcC1QEDR2I!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/AlqWNjeOupOjJlPzI2VbTG.CUyHzcB.hnW5wIaRdW0o!/b/dDABAAAAAAAA&bo=yQK2AckCtgEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/MsNHFUn2mMso7UDUzqXgfphEi7z4gnDOVGjZA26RroU!/b/dDABAAAAAAAA&bo=ugLWAboC1gEDR2I!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/BnyywVMq9dNKuODjt6TNUkeFfk7DlI35SI5uOWS2zqc!/b/dDABAAAAAAAA&bo=uALVAbgC1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/CVX4YJTRREITWV1EhrEn9Ukd8yFAJbsmu5jnTTlgfDU!/b/dDMBAAAAAAAA&bo=yQLEAckCxAEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/0*K5R*CzvH5ZDg24ZOf8S5CDnGmllxPVb0N*shPFylc!/b/dEYBAAAAAAAA&bo=yQJwAckCcAEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/qGf*zeYQrqXxUSTNFKo5U9ozqZB2XPHc6V8jc07KxOk!/b/dDABAAAAAAAA&bo=TwLVAU8C1QEDJwI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/*1pCDquiIpVDuogaqOh1V.oBzefZmXH4LRIG5Zg5oQY!/b/dC0BAAAAAAAA&bo=vgLWAb4C1gEDR2I!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/Kik3NTlTZHYHueKYkZfSOlhv*z4qJj..r.fQvEs*aEM!/b/dIMAAAAAAAAA&bo=3wHVAd8B1QEDJwI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/31YddWSaF8lqoHwzmtoV0Ez13RrkK5SyVVu1p7DPwK0!/b/dDEBAAAAAAAA&bo=yQJ1AckCdQEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/uLJLlUFtfUhTccLySGt7.Zgh2IURhLrNXRlg2lOUGF0!/b/dDEBAAAAAAAA&bo=twLVAbcC1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/*O69lrHWZOQA3cHSbmvZJ6*ctjyqf1ldTaIoa*Jz5TU!/b/dEIBAAAAAAAA&bo=yQKdAckCnQEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/0Cn*uGdLQkaDKxVN5GUKF6SZxuh.AWhKA.6hd2iARLE!/b/dDMBAAAAAAAA&bo=uALVAbgC1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/tprn1lfWm8IPglF9TcSrgYAGgii8zqTUtcTARfS1HIs!/b/dAgBAAAAAAAA&bo=bwLVAW8C1QEDNxI!&rf=viewer_4" width="40%" height="40%" />
</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/vR87Nu35M7uJWN*KrFxWx1kTvUdhHd2kMoevLgSMUAE!/b/dDEBAAAAAAAA&bo=yQJnAckCZwEDJwI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/gA0lkW2WrcN74tYvwXzq*NkCxA8RIUcUDulw*J*yyAA!/b/dC4BAAAAAAAA&bo=YgLWAWIC1gEDR2I!&rf=viewer_4" width="40%" height="40%" />

<img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/vJZNyBtXy1PzBZNJrUg2tVgA*PVa0teVmC5Z*j7U9So!/b/dDMBAAAAAAAA&bo=yQJzAckCcwEDNxI!&rf=viewer_4" width="40%" height="40%" />

</figure><figure class="half">
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/cfAihJGiWT440nNidF.jhoCOv*HD83VOlbNHc7.ROyo!/b/dDABAAAAAAAA&bo=yQJcAckCXAEDNxI!&rf=viewer_4" width="40%" height="40%" />
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/AazS3z5nNJHJHjw5qxlYQXA6VMc2WjLUWVntUtjlNmw!/b/dDABAAAAAAAA&bo=bwHWAW8B1gEDJwI!&rf=viewer_4" width="40%" height="40%" />
</figure>

# 三、存在的问题
1. 工作站显卡性能有限，训练时经常因为显存不足而退出训练；
<center>
    <img src="http://m.qpic.cn/psb?/V13EpJbL1fHfxN/KhJTEVGPqPrE.6cf8JXcyJr.8yqjaf73INMJlkcHa90!/b/dDMBAAAAAAAA&bo=rANxAAAAAAADJ9w!&rf=viewer_4">
</center>
2. 分类错误，一共191类坦克，但神经网络将所有坦克分类为‘5’或‘51’。

# 四、问题分析及初步解决思路
1. 暂无采购显卡的预算，所以目前的思路有2个。（1）进一步阅读相关文献，寻找较小的网络；（2）进一步调小Batch_size,dimension等参数。

2. 坦克与background的区分还是挺明显的，但是坦克的类间差距不明显，加深提出特征向量的CNN的层数或许能改善分类效果。


---
> 20180710 V1
# 一、项目任务：
基于ssd_inception_v2_coco模型初步实现检测过程，并在训练及测试的过程中，寻找原方案的不足点，并加以改进。

# 二、初步检测效果
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/C.px9OpYf0ABwVyFlBmorgp35ICH74rJU5EDwswwQGg!/b/dGEBAAAAAAAA&bo=UALVAVAC1QEDByI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/lb6sEz6vXaYJQqzCkTkhGmA*iSe*x5rekYLeTGWy0Q0!/b/dDMBAAAAAAAA&bo=bwLVAW8C1QEDJwI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/CHnBmmdCjQwqT2enA0fDGSpkY6qy657P89y.nx4w2lo!/b/dEUBAAAAAAAA&bo=zwKsAc8CrAEDR2I!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/GN0eYBLX94zMej.toL58wyLa4U0b39Pm*PPEXofsnVo!/b/dEcBAAAAAAAA&bo=yQKbAckCmwEDNxI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/uUgS*DJ2y20o*RLrzTG5Ow6ss7aFPqGDpDphi*7lku0!/b/dC0BAAAAAAAA&bo=QALWAUAC1gEDJwI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/OySh3o*oYK9vFVp*yNM15QC2fhwaHjWpkh94mvdwt*Q!/b/dDMBAAAAAAAA&bo=bwLVAW8C1QEDJwI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/4ljHw1yAYobakO4GIHEg*p3VPDbIy5qmdk6ZmWtKlpM!/b/dEABAAAAAAAA&bo=0gKRAdICkQEDJwI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/d7phZ8Gahh0nicZJKW0xD.KF0aipRU88M4CFYRRxq1U!/b/dFoAAAAAAAAA&bo=mQLVAZkC1QEDJwI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/Lc7yotKZSmyxAkl1kmJTJ3LklHpGqm5LpZE0v*6Ydz4!/b/dDABAAAAAAAA&bo=swLVAbMC1QEDFzI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/YwWmimMInAmahn29ARQieUNZCZ0nPgdy1g9Ytz.Cwm4!/b/dEIBAAAAAAAA&bo=FALVARQC1QEDNxI!&rf=viewer_4" width="50%" height="50%" />
<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/2ViqBfkS6lVWt2rniyrmt*eBw0X4ZZ1MRD0hdEw9IaA!/b/dDABAAAAAAAA&bo=6wHVAesB1QEDJwI!&rf=viewer_4" width="50%" height="50%" />

# 三、分析
1. 由于训练还没有结束，效果并不算理想。虽然对于大多数test图片，ssd_inception_v2_coco能够检测出目标，但bounding box的精修还很不到位，未能准确的框出目标；
2. 模型未能给出坦克的分类，我需要检查一下是哪里出了问题；
3. GTX1050ti的显存不够用，尽管把batch_size从24削减到6后，还是经常提示显存不够用。在batch_size=6时，训练一个batch大约需要7.4s，还是挺慢的。

# 四、下一步计划
1. 就目前拥有的平台而言，没有能力对大规模网络进行训练，所以下一步将寻找一些规模较小的网络来进行优化。
2. ssd训练完成后，会通过测试集给出mAP。
