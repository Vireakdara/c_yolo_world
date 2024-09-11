## Fast R-CNN
This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection which is an improvement on the previous R-CNN (Region-based Convolutional Neural Networks) model.
The main goal of Fast R-CNN is to reduce the computational cost of applying Convolutional Neural Networks (CNNs) to object detection tasks and improve the efficiency of the model.

## Structure of Fast R-CNN

Unlike R-CNN, which applies the CNN on each region of interest (RoI) separately, Fast R-CNN uses a more efficient approach. It applies the CNN on the entire image only once and then extracts the features for each RoI from the feature map.

The architecture of Fast R-CNN can be summarized in three main stages:

### Convolutional Layers: 
The CNN takes an entire image and a set of Rols as input. It outputs a convolutional features map. 

### Rol Pooling Layer:
From the features map, it extracts a fixed-length feature vector from each Rol using a Rol pooling layer. This feature vector is then mapped to a smaller feature map, which has a fixed size, regardless of the Rol's size.

### Rol Pooling Layer:
The feature vectors are fed into a sequence of fully connected (fc) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K+1 object classes plus a catch-all "background" class, and another layer that outputs four real-valued numbers for each of the K object classes. Each set of four values encodes refined bounding-box positions for one of the K classes.

![image](https://github.com/WeReFxdu/22-Lightweight-Object-Detection-Dara/assets/76716485/e760fc5c-c597-482f-8616-bfdc61c05955)

## Advantages of Fast R-CNN
Fast R-CNN has several advantages over the previous R-CNN:

### Speed and Efficiency:
Fast R-CNN is much faster than R-CNN as it only requires a single pass through the CNN.

### Training:
Unlike R-CNN, which requires the expensive step of training three different models (the CNN to extract features, the classifier to classify objects, and a regression model to tighten bounding boxes), Fast R-CNN can be trained end-to-end in a single stage.

### Accuracy:
Fast R-CNN also improves the accuracy of detections by including bounding box regression in the multi-task loss.

## Disadvantages of Fast R-CNN
However, Fast R-CNN still relies on selective search to generate proposal regions, which could be a bottleneck in terms of speed. This was addressed in the later Faster R-CNN model, which introduced a Region Proposal Network (RPN) to replace the selective search.
