#YOLO 梳理
---
YOLO是<2016>提出的一种用于目标检测的算法，历经几年的完善，从最初的YOLO到现在的YOLOv3，效果提升了很多，但是原始的理念并没有改变，就是直接从原图上卷积出来目标的类别、位置。这种端到端的做法使得YOLO在保持较高准确率的同时，极大地提高了检测速度。而且YOLO在AP.5上的准确度很高，也就是YOLO善于识别类别，但是对于位置的准确度不高，所以在我们目前缺陷检测的应用上使用较为合适，但是需要对于数据集做很好的处理工作。

## 1.YOLOv1

	layer     filters    size              input                output
	   0 conv     64  7 x 7 / 2   448 x 448 x   3   ->   224 x 224 x  64 0.944 BF
	   1 max          2 x 2 / 2   224 x 224 x  64   ->   112 x 112 x  64 0.003 BF
	   2 conv    192  3 x 3 / 1   112 x 112 x  64   ->   112 x 112 x 192 2.775 BF
	   3 max          2 x 2 / 2   112 x 112 x 192   ->    56 x  56 x 192 0.002 BF
	   4 conv    128  1 x 1 / 1    56 x  56 x 192   ->    56 x  56 x 128 0.154 BF
	   5 conv    256  3 x 3 / 1    56 x  56 x 128   ->    56 x  56 x 256 1.850 BF
	   6 conv    256  1 x 1 / 1    56 x  56 x 256   ->    56 x  56 x 256 0.411 BF
	   7 conv    512  3 x 3 / 1    56 x  56 x 256   ->    56 x  56 x 512 7.399 BF
	   8 max          2 x 2 / 2    56 x  56 x 512   ->    28 x  28 x 512 0.002 BF
	   9 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
	  10 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
	  11 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
	  12 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
	  13 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
	  14 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
	  15 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
	  16 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
	  17 conv    512  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 512 0.411 BF
	  18 conv   1024  3 x 3 / 1    28 x  28 x 512   ->    28 x  28 x1024 7.399 BF
	  19 max          2 x 2 / 2    28 x  28 x1024   ->    14 x  14 x1024 0.001 BF
	  20 conv    512  1 x 1 / 1    14 x  14 x1024   ->    14 x  14 x 512 0.206 BF
	  21 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024 1.850 BF
	  22 conv    512  1 x 1 / 1    14 x  14 x1024   ->    14 x  14 x 512 0.206 BF
	  23 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024 1.850 BF
	  24 conv   1024  3 x 3 / 1    14 x  14 x1024   ->    14 x  14 x1024 3.699 BF
	  25 conv   1024  3 x 3 / 2    14 x  14 x1024   ->     7 x   7 x1024 0.925 BF
	  26 conv   1024  3 x 3 / 1     7 x   7 x1024   ->     7 x   7 x1024 0.925 BF
	  27 conv   1024  3 x 3 / 1     7 x   7 x1024   ->     7 x   7 x1024 0.925 BF
	  28 Local Layer: 7 x 7 x 1024 image, 256 filters -> 7 x 7 x 256 image
	  29 dropout       p = 0.50               12544  ->  12544
	  30 connected                            12544  ->  1715
	  31 Detection Layer
	forced: Using default '0'
	Total BFLOPS 40.155

Loss函数：  
![yolo_v1_loss](yolo/yolo_v1_loss.png)

其中S=7，B=3，C=20，Ⅱ表示是否存在目标。

### 1.1 主要思路
从图像一路卷积，然后直接回归得到目标的类别、位置。  
将特征图分为S×S的小区域，每个小区域会负责预测类别及包含这一类的可能概率。  
以每个小区域为中心坐标的限定区域，负责预测B个框，每个框有c,x,y,w,h 5个参数需要预测。  
这样最后的全连接层一共需要S×S×(B*5+C)个参数输出。
框的位置是全连接里面连接出来的，可以在loss函数计算公式里面看到。

### 1.2 优缺点
优点: 

* one-stage的方式，速度快，不用分块训练，参数可以一块训练优化。
* 使用全图信息进行训练，背景错误比较少

缺点:

* 对于小物体检测效果较差
* 对于靠的较近的物体预测较差，这个是模型设计好了就可以预知的缺陷
* 同一类物体出现新的长宽比
* 定位误差较大

## 2.YOLOv2
	layer     filters    size              input                output
	   0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32 0.299 BF
	   1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32 0.006 BF
	   2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64 1.595 BF
	   3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64 0.003 BF
	   4 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
	   5 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
	   6 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
	   7 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x 128 0.001 BF
	   8 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	   9 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  10 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  11 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 256 0.001 BF
	  12 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  13 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  14 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  15 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  16 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  17 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 512 0.000 BF
	  18 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  19 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  20 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  21 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  22 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  23 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024 3.190 BF
	  24 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024 3.190 BF
	  25 route  16
	  26 conv     64  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  64 0.044 BF
	  27 reorg              / 2    26 x  26 x  64   ->    13 x  13 x 256
	  28 route  27 24
	  29 conv   1024  3 x 3 / 1    13 x  13 x1280   ->    13 x  13 x1024 3.987 BF
	  30 conv    425  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 425 0.147 BF
	  31 detection
	mask_scale: Using default '1.000000'
	Total BFLOPS 29.475

改进的地方:

1. Batch Normalization. 改善了网络中间层的输入分布变换太大的情况，减少内部协方差的影响，加速迭代收敛的过程.
2. High Resolution. 分类器的分辨率从224提升到了448,后来有所修改.
3. Convolutional with Anchor Boxes. 借鉴了R-CNN的思路,将RPN移植到网络架构里,提升了recall,同时对每个archor box预测类型和目标参数.
4. Dimension Clusters. 把训练集的框kmans聚类后得到一些先验的、效果较好的框的参数.
5. Driect location prediction. 把FPN的位置预测做了调整,限制了archor box可能出现的位置和大小.
6. Fine-Grained Features. 直接将中间某一层的输出拿过来和后面的做了融合,能够反映更多的细微特征,有利于检测小物体.
7. Multi-Scale Training. 输入的图像size从320~608之间变化,以32为步长,没有全连接层固定参数就可以训练不同输入尺度的图像.

## 3.YOLOv3
	layer     filters    size              input                output
	   0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32 0.299 BF
	   1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64 1.595 BF
	   2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32 0.177 BF
	   3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64 1.595 BF
	   4 Shortcut Layer: 1
	   5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128 1.595 BF
	   6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
	   7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
	   8 Shortcut Layer: 5
	   9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64 0.177 BF
	  10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128 1.595 BF
	  11 Shortcut Layer: 8
	  12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256 1.595 BF
	  13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  15 Shortcut Layer: 12
	  16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  18 Shortcut Layer: 15
	  19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  21 Shortcut Layer: 18
	  22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  24 Shortcut Layer: 21
	  25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  27 Shortcut Layer: 24
	  28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  30 Shortcut Layer: 27
	  31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  33 Shortcut Layer: 30
	  34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	  35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	  36 Shortcut Layer: 33
	  37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512 1.595 BF
	  38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  40 Shortcut Layer: 37
	  41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  43 Shortcut Layer: 40
	  44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  46 Shortcut Layer: 43
	  47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  49 Shortcut Layer: 46
	  50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  52 Shortcut Layer: 49
	  53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  55 Shortcut Layer: 52
	  56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  58 Shortcut Layer: 55
	  59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  61 Shortcut Layer: 58
	  62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024 1.595 BF
	  63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  65 Shortcut Layer: 62
	  66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  68 Shortcut Layer: 65
	  69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  71 Shortcut Layer: 68
	  72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  74 Shortcut Layer: 71
	  75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512 0.177 BF
	  80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024 1.595 BF
	  81 conv    255  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 255 0.088 BF
	  82 yolo
	  83 route  79
	  84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256 0.044 BF
	  85 upsample            2x    13 x  13 x 256   ->    26 x  26 x 256
	  86 route  85 61
	  87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256 0.266 BF
	  88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256 0.177 BF
	  92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512 1.595 BF
	  93 conv    255  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 255 0.177 BF
	  94 yolo
	  95 route  91
	  96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128 0.044 BF
	  97 upsample            2x    26 x  26 x 128   ->    52 x  52 x 128
	  98 route  97 36
	  99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128 0.266 BF
	 100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	 101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	 102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	 103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128 0.177 BF
	 104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256 1.595 BF
	 105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255 0.353 BF
	 106 yolo
	Total BFLOPS 65.864

改进的地方:

1. 从其他网络结构借鉴了很多有利的做法,但是没有大的改进.
2. 从fpn借鉴了多尺度预测的概念,将低层和高层信息融合并预测.

其他问题:

1. 目前的map计算方法有问题,COCO太偏重于框的准确程度,其实IOU在0.5左右已经是一个较好的指标了.
2. 现在的map计算存在误差,有些情况下完全不能分辨.
3. bounding box是一种很差的方法.

## 4.DOLO

## 5.其他和YOLO相关的算法