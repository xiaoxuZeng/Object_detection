# 一、Related terms
1. Non-Maximum Suppression：根据score和box的坐标信息，从中找到置信度比较高的bounding box。首先，然后根据score进行排序，把score最大的bounding box拿出来。计算其余bounding box与这个box的IoU，然后去除IoU大于设定的阈值的bounding box。然后重复上面的过程，直至候选bounding box为空。说白了就是我要在一堆矩阵里面找出一些局部最大值，所以要把和这些局部最大值所代表矩阵IoU比较大的去除掉，这样就能得到一些权值很大，而且IoU又比较小的bounding box。
2. end-to-end方法：一端输入我的原始图像，一端输出图像中所有目标的位置和目标的类别。
3. 候选区域（anchor）
4. SPP(spatial pyramid pooling)：将不同尺寸的输入resize成为相同尺寸的输出。
5. IoU：交并比
# 二、References
[【目标检测】Faster RCNN算法详解](https://blog.csdn.net/shenxiaolu1984/article/details/51152614)

[马塔的回答：faster rcnn中rpn的anchor，sliding windows，proposals？](https://www.zhihu.com/people/liu-ke-91-47/answers)

[非极大抑制（Non-Maximum Suppression](https://www.cnblogs.com/makefile/p/nms.html)
