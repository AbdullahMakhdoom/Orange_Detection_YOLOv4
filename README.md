# Orange_Detection_YOLOv4
Custom trained YOLOv4 in python for orange detection and yield estimation.

## Downloading & Configuring YOLOv4 for Training on Custom Orange Dataset
1.    Download or clone the repository from this [link](https://github.com/AlexeyAB/darknet.git) .
2.    Unzip the contents of the repository.
3.    Make the following changes :-

*    Open "darknet-master" folder which we have just unzipped. Go to cfg folder.Make a copy of the file yolo4-custom.cfg now rename the copy file to yolo-obj.cfg.
*    Open the file yolo-obj.cfg then :- (or skip this step by downloading yolo-object/yolo-obj.cfg)

      a.   change max_batches to (classes*2000), in our orange dataset classes = 1, so max_batches = 2000.

      b.   change the line steps to (0.8 * max_batches ,0.9 * max_batches), in our case it will be steps = 1600, 1800.

      c.   set network size width=416 height=416.

      d.   change line classes=80 to your number of objects in each of 3 yolo layers. which is just 1.

      e.    change [filters=255] to filters=(classes + 5)x3 in the 3 convolutional  layer immediately before each 3 yolo layers. In our scenario, filters=18.


4. Download the pre trained weights from the link [ yolo4.conv.137 ](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view) and save it in the darknet-master folder.
5. Open wordpad and type the name of each object in separate lines and save the file as obj.names in darknet-master->data folder. In our case, just orange. Make sure there are not spaces at the end of file.
6. Create file obj.data in the folder darknet-master->data, containing the given text (replace classes = number of objects)
classes= 1
train  = data/train.txt
names = data/obj.names
backup = backup/
7. Create a folder in the directory darknet-master->data named obj now inside this obj folder put all your images and the respective txt files you got from labeling in step1
8. Now we have to create a train.txt file.This file directs to all training images as shown in the below picture.The easiest way to achieve this is store all images in a folder in computer open command prompt navigate to the folder using ‘cd’ and type command ‘ls’ if in linux and ‘dir’ if in windows.This will display all image names copy that and paste in text file and add ‘data/obj/images/’ to each line for this Find and replace option could be used.The train.txt file is stored in the darknet-master->data folder.

 ```
data/obj/images/Img1.png
data/obj/images/Img2.PNG
data/obj/images/Img3.PNG
data/obj/images/Img4.PNG
... 
```

9. In the darknet-master folder open Makefile in wordpad and change GPU=1,CUDNN=1,OPENCV=1 as shown in the following picture.This is done to make the training on GPU.

```
GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0 # ZED SDK 3.0 and above
ZED_CAMERA_v2_8=0 # ZED SDK 2.X
```

## Inference using GUI 
1. Create virtual environment (recommneded)
2. Clone the repo or download the repo's .zip file.
3. Install necessary packages by running "pip install -r requirements.txt" on command line
4. Download custom (trained on custom orange dataset) yolo weights from https://drive.google.com/file/d/1Mo6yHlriemyFqWsulWO2az3QINwsTX6i/view?usp=sharing.
5. run "python3 main.py" for starting GUI
6. Browse the images to detect oranges
7. Provide path to base folder (which is yolo-orange) containing .cfg, .name  and .weights files.
8. Adjust Confidence and Threshold for YOLOv4 detector
9. Press OK to get detectors result.
