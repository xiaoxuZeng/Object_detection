from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']



def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join('train_data',data['folder'], 'image', data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bounding_box']['x_left_top']) / width)
      ymin.append(float(obj['bounding_box']['y_left_top']) / height)
      xmax_ = float(obj['bounding_box']['x_left_top'])+float(obj['bounding_box']['width'])
      ymax_ = float(obj['bounding_box']['y_left_top'])+float(obj['bounding_box']['height'])
      xmax.append(float(xmax_) / width)
      ymax.append(float(ymax_) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']]) #labelmap
      truncated.append(0)#truncated=0
      poses.append(b'Unspecified')

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example

def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict('/home/zju/models/research/object_detection/Det_datasets/pascal_label_map.pbtxt')

  for i in range(4,7):
      path = os.path.join(data_dir,'train_data','0_1','xml', '1_'+str(i) + '.xml')
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)

      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      print(data)
      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
