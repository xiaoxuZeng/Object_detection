> You Only Look Once: Unified, Real-Time Object Detection

<img src="http://m.qpic.cn/psb?/V13EpJbL3HbDX9/xq0va7FqgCiK9Vl9HfL6FfNdMpmSaoD7Bs10PN.1CU8!/b/dAgBAAAAAAAA&bo=kAKYAZACmAEDByI!&rf=viewer_4" width="40%" height="40%" />

1. 提取特征向量
2. 将特征向量分成S*S个单元
3. 每个单元做2个操作
    1. 每个单元预测B个bounding box并给出每个bounding box的置信度(一共是B*（4+1）个参数)
    2. 每个单元预测k个概率（一共k类）
4. 将每个单元的bounding box的置信度和每个单元预测的k个概率乘起来，就得到了每个bounding box包含某类物体的概率。每个单元一共有kB个概率。
5. 经过non-max suppress，剔除无用框，得到最终结果。
