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
3. 相关数据在Github文件中
### 相关类含义：

class| mean 
---|---
class1 | 轧制重皮
class2 | 重皮
class3|轧伤成品
class4|轧制凹点成品
class5|轧制条痕原料
class6|孔洞
class7|孔洞镀后
class8|成品划伤
class9|稳定辊划伤
class10|针孔漏镀
class11|凹印
class12|针孔


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
# 四、结果分析
训练后的模型取得了一定效果，证明”采用迁移学习的方式对相关经典模型进行微调“的思路是可行的！
# 五、存在的问题及解决思路
## 5.1 问题
1. 轧制重皮、重皮、轧伤成品、稳定辊划伤这四类瑕疵检测效果不佳
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/Fn.sEBDP4WncYIr7oSXOSmeemwAnbjBc5jOuTpfEoWs!/b/dFoAAAAAAAAA&bo=yQJYAckCWAEDByI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/DJCgSp5bnkXNPynVRXB.qKVD1eBO7oL9dcv5pKkFqhQ!/b/dDEBAAAAAAAA&bo=FAHVARQB1QEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/9bP5XBZxa7VPWEbmiwFCldsiHn7c0XuulD*oYwbENBk!/b/dDABAAAAAAAA&bo=yQJuAckCbgEDFzI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/spuIDJ7mRR4QtgQgPsfQTIgef6wWer0LI84NAB1TyXc!/b/dGcBAAAAAAAA&bo=vQHVAb0B1QEDFzI!&rf=viewer_4)
2. 开放场景性效果很一般

![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/lgFtAtNL.Ob1vFhJL9*XszTkG*SndzMJXhL7ke1dcG0!/b/dDIBAAAAAAAA&bo=dQLWAXUC1gEDJwI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/lj6jaOfSuEcYY74lCKomfMGe2Jiywe2UWykglw3j8Yg!/b/dEABAAAAAAAA&bo=dQLWAXUC1gEDNxI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/oZyieYIsXZaRoqAaFDtkg6.9ZDHKBfBalz8hGy2puJw!/b/dC4BAAAAAAAA&bo=dQLWAXUC1gEDNxI!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/ACOSgWsyQicvSbS8Eqe0tMef4g36sQapL4QtsifUPXY!/b/dDABAAAAAAAA&bo=zwKbAc8CmwEDNxI!&rf=viewer_4)
## 5.2 问题分析
1. 训练数据实在是太少，每个类只有一张，最后采用增强数据的方式，对原始图片进行裁剪、旋转，最终得到62张训练图片；
2. 这62张训练图片分布不均匀，包含‘轧制重皮、重皮、轧伤成品、稳定辊划伤这‘四类瑕疵的图片十分少，这直接造成训练后的模型对这四类瑕疵检测效果不佳；
3. 拿到的数据是在开放环境下拍摄的，关照等拍摄环境差异巨大；
4. 我的标签不专业，图片标记存在漏标甚至错标的情况。
## 5.3解决思路
1. **获取更多、更专业数据（最重要）**
    
    要求：    
    1.   在特定环境下拍摄图片：同样的光照、同样的角度等；
    2.   请专业人员对图片进行标记，尽量减少漏标及错标的情况。
2. 选择更加强大的模型进行迁移学习；
3. 根据具体情况，设计一个神经网络进行训练。

# 六、下一阶段研究方向
进一步阅读相关文献，寻找更加强大的模型。争取能根据具体情况，设计一个神经网络进行训练。
