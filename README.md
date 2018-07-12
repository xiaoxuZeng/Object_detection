## 一、[编译proto文件](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
```
# From tensorflow/models/research/
1. protoc object_detection/protos/*.proto --python_out=.
2. export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# test
3. python3 object_detection/builders/model_builder_test.py
```
## 二、[训练新的模型](https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_5/README.md)
```
1. cd models/research/object_detection
2. python3 dataset_tools/create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
3. python3 dataset_tools/create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
4. cp data/pascal_label_map.pbtxt voc/
# 下载模型文件http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
并解压，解压后得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 个文件。在voc文件夹中新建一个 pretrained 文件夹，并将这5个文件复制进去。 
5. cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
6. 修改voc.config
7. python3 train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
8. tensorboard --logdir voc/train_dir/
9. python3 export_inference_graph.py \ 
--input_type image_tensor \ 
--pipeline_config_path voc/voc.config \
--trained_checkpoint_prefix voc/train_dir/model.ckpt-7622 \
--output_directory voc/export/
```
## 三、遇到的坑
1. ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

pytho3兼容性问题
[参见](https://github.com/tensorflow/models/issues/3752)
```
I believe it is the same Python3 incompatibility that has crept up before (see #3443 ). The issue is with models/research/object_detection/utils/learning_schedules.py lines 167-169. Currently it is

rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries),
                                      [0] * num_boundaries))
Wrap list() around the range() like this:

rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),
                                      [0] * num_boundaries))
and you should be good to go. Mine is off and training.

@jwnsu make sure you change your pipeline.config from:

        manual_step_learning_rate {
          initial_learning_rate: 0.00001
          schedule {
            step: 0
            learning_rate: .0003
          }
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
to the new slightly different:

        manual_step_learning_rate {
          initial_learning_rate: 0.0003
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
2. [错误：tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.](http://blog.sina.com.cn/s/blog_92d2c5e10102xhxj.html)
分析：出现这个错误一般是GPU内存耗尽。
解决办法：运行程序之前，先运行export CUDA_VISIBLE_DEVICES=0

3. ImportError: No module named '_tkinter', please install the python3-tk package
[解决方法](https://blog.csdn.net/qq_18293213/article/details/74483516)：
```
1. sudo apt-get install python3-tk 
2. sudo apt-get install -f 
```
4. TypeError: 'range' object does not support item assignment
```
In python3 range is a generator object - it does not return a list. Convert it to a list before shuffling.
```
## 四、检测结果
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/jsu8LOoB6UK628tijd02h3aO5kvAa45rfZwzOwzNPGE!/b/dFcAAAAAAAAA&bo=9AHVAQAAAAADBwM!&rf=viewer_4)
![image](http://m.qpic.cn/psb?/V13EpJbL3HbDX9/Rw*VMOSIx.sidQX8AVbZN6wuGLUYYdKD.3RUNcTCxKk!/b/dC8BAAAAAAAA&bo=TgHVAQAAAAADB7k!&rf=viewer_4)

## 五、参考资料
1. [tensorflow 轻松实现自己的目标检测](https://blog.csdn.net/wangjian1204/article/details/79124018)
2. [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
3. [Object detection with TensorFlow - O'Reilly Media](https://www.oreilly.com/ideas/object-detection-with-tensorflow)
