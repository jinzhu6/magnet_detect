#YOLO 梳理
---
YOLO是<学校><人><时间>提出的一种用于目标检测的算法，历经几年的完善，从最初的YOLO到现在的YOLOv3，效果提升了很多，但是原始的理念并没有改变，就是直接从原图上卷积出来目标的类别、位置。这种端到端的做法使得YOLO在保持较高准确率的同时，极大地提高了检测速度。而且YOLO在AP.5上的准确度很高，也就是YOLO善于识别类别，但是对于位置的准确度不高，所以在我们目前缺陷检测的应用上使用较为合适，但是需要对于数据集做很好的处理工作。

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

改进的地方：


## 3.YOLOv3

## 4.DOLO

## 5.其他和YOLO相关的算法