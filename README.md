

[![Build Status](https://travis-ci.com/rij12/YOPO.svg?token=an4QsGxZQ9sn7osFx53B&branch=yopo)](https://travis-ci.com/rij12/YOPO) 


# YOPO

Is an adaptation of the YOLOv1 algorithm and made using Darkflow. Where by a new framework is added that is based on the original YOLOv1 framework in Darkflow. 


YOPO is split into two parts: the pre-processing and the YOPO network. 


### Getting started

1. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip3 install -e .
    ```

2. Install with pip globally
    ```
    pip3 install -r requirements.txt --user
    ```

### YOPO preprocessing


The preprocessing part requires:

1. The MPII dataset images which can be found here: http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
2. The MatLab data file containing the metadata about the dataset here (default location in YOPO_preprocessing/data): http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
3. The config entries to be set your your configuration(default location:  YOPO/YOPO_preprocessing/src/main/conig.py), for example set the location of the MPII dataset images and MatLab data file. 
    
### YOPO Network 

1. The project was developed with pycharm so to run it from the command you must export the path for example:

    ```
    export PYTHONPATH=/home/USER/git/YOPO
    ```
2. Just build the Cython extensions in place. 

    ```
    python3 setup.py build_ext --inplace

    ```
    
3. Each limb is defined in a text file labels.txt

4. The preprocessing must have been completed with images and label folder created(default: YOPO/YOPO_preprocessing/data/darkflow). 

5. Training the network:

    For any addtional options use:
    
     ```
     python3 ./flow --h
     ``` 

    * --model is the network configuration file, that defines that network.
    * --load is used to supply weights to use as a starting point, as discussed in the report YOPO uses the original weights.
    * --annotations The file path with a folder containing the labels for the training images
    * --dataset The file path with a folder containing the images that the network will train on.
    * --epoch How many the epoch should be executed before the network stops. 
    * --gpu Optional flag but recommended. The percentage of GPU usage that Darkflow is allows to use for training the network.  
    * --summary Optional flag for the TensorBoard event file to be save for later use in TensorBoard. 
    * --batch defines the amount of image that be processed through the network at once. If you have a small amount of RAM or VRAM use a low batchsize. 
    
  
    ```
    python3 ./flow --model NETWORK_CONFIG_PATH --load YOLO_WEIGHTS_PATH --train --annotation LABELS_PATH --dataset DATASET_PATH --epoch 20000 --gpu 0.9 --summary OUTPATH --batch 8
    ```


The weight outputted from the training of the network are save in a folder called ckpt that will be create during training. 
   
   
6. Testing the network
 
    Where
    
    * --imgdir is the image that you would like to train on
    * --model is the network configuration file 
    * --load is the weight file that was outputted from the network, -1 can be used to load the last outputted weight from the network. 
    * --threshold the minimum probability a box must have before it's shown on the output images

    ```
    python3 ./flow --imgdir sub_set/images --model /home/richard/git/YOPO/cfg/yopo.cfg --load -1 --threshold 0.1

    ```
    



## Acknowledgments

This project is an adapted folk of Darkflow (https://github.com/thtrieu/darkflow), with the attempt of creating a human pose estimation system. 

YOLO papers, Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/).
    






