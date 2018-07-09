
import hashlib
import io
import os

import tensorflow as tf
from xml.etree.ElementTree import ElementTree, Element


def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8", xml_declaration=False)

def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True



def read_examples_list(path):
    """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
        image_info = [line.strip().split(' ')[0] for line in lines]
        annotation_info = [line.strip().split(' ')[1] for line in lines]
    return (annotation_info)



def main(_):
    txt_path = os.path.join('Det_datasets', 'train_list' + '.txt')
    examples_list = read_examples_list(txt_path)
    for idx, example in enumerate(examples_list):
        label_path = os.path.join('Det_datasets', example)
        # 1. 读取xml文件
        tree = read_xml(label_path)
        # 2. 输出到结果文件
        write_xml(tree,label_path)

if __name__ == '__main__':
  tf.app.run()
