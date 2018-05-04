import glob
import time

from YOPO_preprocessing.src.main.config import config
from YOPO_preprocessing.src.busniess.generate_limb_bbox_darkflow import generate_limb_data
from YOPO_preprocessing.src.utils.util import load_matlab_data, darkflow_sort_images

images = glob.glob("{}*jpg".format(config['IMAGE_PATH']))

'''
Process chain 

1. Take the MatLab data files and convert them to python dict 
2. Split data into train set and test (default 80:20)
3. Take image set and generate ground truths for Darkflow 
4. Sort images and XML files into correct folder ready for training  

'''

if __name__ == "__main__":

    if len(images) == 0:
        print("ERROR: CANNOT FIND IMAGE DATA!")
        exit(-1)

    # Define test and train sets:

    trainSet = int(len(images) * 0.8)
    testSet = int(len(images) * 0.2)
    train = images[:trainSet]
    test = images[trainSet:]

    # Load image meta data.
    data = load_matlab_data()

    start_time = time.time()

    # Generate Limb data and sort them into folder read for training the network

    # Train dataset
    generate_limb_data(image_file_path_list=train, image_metadata=data, train=True)
    darkflow_sort_images(train=train)

    # Test dataset
    generate_limb_data(image_file_path_list=test, image_metadata=data, train=False)
    darkflow_sort_images(train=False)

    print("Finished in %s seconds " % int(time.time() - start_time))
