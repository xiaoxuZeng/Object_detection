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

if __name__ == "__main__":
    # 1. 读取xml文件
    tree = read_xml("/home/zju/models/research/object_detection/Det_datasets/train_data/0_1/xml/1_8.xml")
    # 2. 输出到结果文件
    write_xml(tree, "/home/zju/models/research/object_detection/Det_datasets/train_data/0_1/xml/1_9.xml")