import glob
import time

from YOPO_preprocessing.src.main.config import config
from YOPO_preprocessing.src.busniess.generate_limb_bbox_darkflow import generate_limb_data
from YOPO_preprocessing.src.utils.util import load_matlab_data, darkflow_sort_images

images = glob.glob("{}*jpg".format(config['IMAGE_PATH']))

'''
Process chain 

1. Take the MatLab data files and convert them to python dict 
2. Take image set and generate ground truths in the YOLO or Darkflow format 
3. Output bounding boxes using YOLO ground truth information.
4. Move Images and text files into folder ready for the network.

'''

if __name__ == "__main__":

    if len(images) == 0:
        print("ERROR: CANNOT FIND IMAGE DATA!")
        exit(-1)

    # Load image meta data.
    data = load_matlab_data()

    start_time = time.time()

    # Generate Limb data and sort them into folder read for training the network
    generate_limb_data(image_file_path_list=images, image_metadata=data, train=True)
    darkflow_sort_images()

    print("Finished in %s seconds " % int(time.time() - start_time))
