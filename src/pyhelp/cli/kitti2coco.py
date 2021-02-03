"""
    This script convert a kitti 2D detection dataset into a "custom" dataset in mmdetection

    KITTI dataset:

    -training
        --image_2
            000000.png
            000001.png
            ...

        --label_2
                #Values    Name      Description
            ----------------------------------------------------------------------------
            1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                                'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                'Misc' or 'DontCare'
            1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                                truncated refers to the object leaving image boundaries
            1    occluded     Integer (0,1,2,3) indicating occlusion state:
                                0 = fully visible, 1 = partly occluded
                                2 = largely occluded, 3 = unknown
            1    alpha        Observation angle of object, ranging [-pi..pi]
            4    bbox         2D bounding box of object in the image (0-based index):
                                contains left, top, right, bottom pixel coordinates
            3    dimensions   3D object dimensions: height, width, length (in meters)
            3    location     3D object location x,y,z in camera coordinates (in meters)
            1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
            1    score        Only for results: Float, indicating confidence in
                                detection, needed for p/r curves, higher is better.

    COCO dataset for detection.
    Json format:
    {
        "info" : {
            # can be an empty dictionary. COCO.info in pycocoAPI will print out all the key-value pair of this dictionary without checking if "info" is in dataset.
        },
        "images" :[image],
        "annotations: [annotation],
        "licenses": [licence], # not nesessary, not directly used in COCO API. 
        "categories": [category]
    }
    image{
        "id": int,
        "width": int, "height": int,
        "file_name": str,
        "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime,
    }
    annotation{
        "id": int, 
        "image_id": int, 
        "category_id": int, 
        "segmentation": RLE or [polygon], # For bounding box detection, omit such field
        "area": float, 
        "bbox": [x,y,width,height], #[top_left_x, top_left_y, w, h]
        "iscrowd": 0 or 1, #ignore or not
    }
    
    Sample usage:
    ```bash
        pyhelp.kitti2coco --kitti_path=<kitti_path>/training(test) --output_file=<output_path>.json --label_split_file=<path/to/splitfile.txt or None or omitted> --output_count
    ```
"""

from fire import Fire
import json
import os
from PIL import Image
import numpy as np
import tqdm
import datetime
from typing import Union, List, Dict, Any

KITTI_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

def read_label_split_file(label_split_file:Union[None, str],
                          num_total_file:int)->List[bool]:
    if label_split_file is None:
        return [True for _ in range(num_total_file)]
    
    selection_mask = [False for _ in range(num_total_file)]
    with open(label_split_file, 'r') as f:
        for line in f.readlines():
            if line.strip():
                index_num = int(line.strip())
                selection_mask[index_num] = True
    return selection_mask


def kitti2COCO(kitti_path:str, 
                 output_file:str,
                 label_split_file:Union[None, str]=None,
                 output_count:int=10):
    """
        transform kitti labels to coco dataset format
        Args:
            kitti_path: path to KITTI training or test set 
            output_path: output file path to json file of the output dataset
    """
    ## Set up KITTI path and objects
    image_path = os.path.join(kitti_path, "image_2")
    label_path = os.path.join(kitti_path, "label_2")

    image_files = os.listdir(image_path)
    label_files = os.listdir(label_path)
    image_files.sort()
    label_files.sort()

    mask = read_label_split_file(label_split_file, len(image_files))
    image_files = [image_files[i] for i in range(len(image_files)) if mask[i]]
    label_files = [label_files[i] for i in range(len(label_files)) if mask[i]]
    
    ## Set up custom path and objects
    if output_count > len(image_files):
        output_count = len(image_files)
    print("The number of output image will be %d" % output_count)

    ## Set up basic info and formats for coco data
    output_json:Dict[str, Any] = {}
    output_json["info"] = {
        "year" : datetime.datetime.now().year,
        "description" : "Transformed from KITTI dataset with kitti_path:{}; output_path:{}, label_split_file:{}, output_count:{}".format(kitti_path, output_file, label_split_file, output_count),
    }
    output_json["images"] = []
    output_json["annotations"] = []
    output_json["categories"] = []
    for index, item in enumerate(KITTI_NAMES):
        output_json["categories"].append(
            {
                "id":index+1,
                "name":item
            }
        )

    global_annotation_id = 0
    for image_file, label_file in tqdm.tqdm(zip(image_files[:output_count], label_files[:output_count])):
        abs_image_path = os.path.join(image_path, image_file)
        abs_label_path = os.path.join(label_path, label_file)

        image_obj = {
            "id":int(image_file.split(".")[0]),
        }

        # read image
        image = Image.open(abs_image_path)
        width, height =image.size
        image_obj["width"] = width
        image_obj["height"] = height
        image_obj["file_name"] = abs_image_path
        output_json["images"].append(image_obj)

        # read label
        with open(abs_label_path, 'r') as file:
            for line in file.readlines():
                splits = line.split(' ')
                cls_ = splits[0]
                if not cls_ in KITTI_NAMES or float(splits[2]) > 2:
                    continue
                label_id = KITTI_NAMES.index(cls_) + 1 # Notice the first index is expected to be background
                
                bbox = [float(splits[i]) for i in range(4, 8)] #[left top right bottom]
                coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                cocodet_label = {}
                cocodet_label["id"] = global_annotation_id
                cocodet_label["category_id"] = label_id
                cocodet_label["image_id"] = image_obj["id"]
                cocodet_label["area"] = coco_bbox[2] * coco_bbox[3]
                cocodet_label["segmentation"] = []
                cocodet_label["bbox"] = coco_bbox
                cocodet_label["iscrowd"] = 0 # as in KITTI evaluation script, KITTI will filter out objects with oclude > 2 even for hard-estimation
                output_json["annotations"].append(cocodet_label)
                global_annotation_id += 1


        # bboxes = np.array(bboxes).reshape(-1, 4)
        # labels = np.array(labels).reshape(-1)

    json.dump(output_json, open(output_file, 'w'))


def main():
    Fire(kitti2COCO)



if __name__ == '__main__':
    Fire(kitti2COCO)
    
