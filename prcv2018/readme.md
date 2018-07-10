# [PRCV 2018](https://prcv-conf.org/2018/comp_list_05?from=singlemessage&isappinstalled=0)
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
3. GTX1050ti的显存不够用，尽管把batch_num从24削减到6后，还是经常提示显存不够用。在batch_num=6时，训练一个batch大约需要7.4s，还是挺慢的。

# 四、下一步计划
1. 就目前拥有的平台而言，没有能力对大规模网络进行训练，所以下一步将寻找一些规模较小的网络来进行优化。
2. ssd训练完成后，会通过测试集给出mAP。
