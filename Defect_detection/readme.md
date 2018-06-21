# 一、要解决的问题
检测镀锌板疵点
# 二、所采用的方案
## 2.1 问题分析：
属于目标检测问题，由于样本量很小，可采用迁移学习的方式对相关经典模型进行微调。
## 2.2 选取经典模型：
因为没什么数据，所以本工程的主要目的在于验证方案可行性，选取十分经典的[faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017](https://arxiv.org/abs/1506.01497)模型进行迁移学习。

### 相关参数：

Speed (ms) | COCO mAP[^1]
---|---
620 | 37
## 2.3具体做法
1. 图片预处理；
2. 标记；
3. 将处理后的图片及标记转换为tfrecord格式；
4. 进行训练；
5. 检测。
## 2.4相关说明
1. 工作环境：GTX1050ti+Ubuntu16+Python3.6+tensorflow(API1.8)
2. 训练步数：48022
3. tfrecord数据


# 三、检测效果
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/THv2pYDQhertIQtYUKGlqMeY9p3NEJTB9jITtyqKV1w!/b/dFoAAAAAAAAA&bo=5gHVAeYB1QEDByI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/4nlXSZrsJJRn4u02wSl0xjZMcpnCYzunkNUeaeKgcIE!/b/dEEBAAAAAAAA&bo=1AHVAdQB1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/mwHMecZPVxr8nkkMkxqRDuPk9CFUsElqY3ocICC1Isw!/b/dFkAAAAAAAAA&bo=zgLKAc4CygEDJwI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/9LsyRkWhqXNveRX*baOEdITjPtOAFPMgvOwJewC46hQ!/b/dDEBAAAAAAAA&bo=twHVAbcB1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/I3Ic*NiMI8JWf3C0OvVlOOrP.E9DwNkcQKzWhNtNzrw!/b/dDMBAAAAAAAA&bo=1gHVAdYB1QEDJwI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/co*85Ng*mioXX5yzzP2yAk1xR5GdusrGaGwrvp5*4Lc!/b/dFoAAAAAAAAA&bo=yQJ2AMkCdgADFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/rmQHGZTqOKDlqZ9d7l4n.iPHHiGgZ*JqYRCwyx4bhPU!/b/dC0BAAAAAAAA&bo=pAHVAaQB1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/rWv0hq.V8fvWUgJDKRIG9Q.2RHGoAINxjXl2.xrKLtc!/b/dC4BAAAAAAAA&bo=yQJ7AckCewEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/r1kLEoLsXjspKafHBCig*kAY8qq*E.OaoveaLqlvgbY!/b/dFUAAAAAAAAA&bo=pgLVAaYC1QEDNxI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/gqe9kbOyM8aaHbNx0SPUS26iXkyGk0D1yjAQlOiYLbI!/b/dDABAAAAAAAA&bo=xgLVAcYC1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/fyubzHa1jWHPN1DyB8OZKmpK*s9vwNt8nyc1NbVhKSA!/b/dC8BAAAAAAAA&bo=XgLVAV4C1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/QcpgyHkxAxr1V5Gx8quNhFCluORbdyxXf1kLPZ0ykf0!/b/dDIBAAAAAAAA&bo=JgLVASYC1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/AweBLLTyDwnSzs9dhopClmCWoBIYAWtHM8PsGoUVm5Y!/b/dC4BAAAAAAAA&bo=7QHVAe0B1QEDJwI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/XGzK93sesEPktHC3tbKKp9exR4DTHXx8v4PQ9FBNehs!/b/dIMAAAAAAAAA&bo=hQHVAYUB1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/zuWW*OUbBp8v5uTKoL7TQDF1*3t62MYQx5wn8OEhYVw!/b/dDIBAAAAAAAA&bo=yQJxAckCcQEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/xHXVF4Yj9D1OL*jSFSy4LNX*NTxKu7xfVgJncqpoXJ0!/b/dDABAAAAAAAA&bo=pAHVAaQB1QEDFzI!&rf=viewer_4)
# 四、存在的问题及解决思路
