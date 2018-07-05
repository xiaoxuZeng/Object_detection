#convert png to jpeg
# -*- coding: UTF-8 -*-//解决出现Non-ASCII character '\xe5' in file的问题
from PIL import Image
import os, sys

def convert(dir,name):
    im = Image.open(dir)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, (0,0),im)
    save_name = str(name)+".jpg"
    bg.save(save_name)

if __name__ == '__main__':
    dict = '/home/zju/Desktop/train_data/'     # where is storing your png imgs.
    for i in range(1,63):
        dir = dict + str(i) + '.png'
        convert(dir,i)
    print("done")
