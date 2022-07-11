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

    Custom dataset for detection.
    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    The `ann` field is optional for testing.
    
    Sample usage:
    ```bash
        pyhelp.kitti2custom --kitti_path=<kitti_path>/training(test) --output_file=<output_path>.json --label_split_file=<path/to/splitfile.txt or None or omitted> --output_count
    ```
"""
from fire import Fire
import json
import os
import cv2
import tqdm
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


def kitti2custom(kitti_path:str, 
                 output_file:str,
                 label_split_file:Union[None, str]=None,
                 output_count:int=10):
    """transform kitti labels to 'custom' dataset of mmdetection

    Args:
        kitti_path (str): path to KITTI training or test set 
        output_file (str): output file path to json file of the output dataset
        label_split_file (Union[None, str], optional): split_file. Defaults to None, all-true.
        output_count (int, optional): set the upper bound of the output data size. Defaults to 10.
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
    output_json = []
    for image_file, label_file in tqdm.tqdm(zip(image_files[:output_count], label_files[:output_count])):
        abs_image_path = os.path.join(image_path, image_file)
        abs_label_path = os.path.join(label_path, label_file)

        json_object:Dict[str, Any] = dict()
        json_object['filename'] = abs_image_path

        # read image size
        image = cv2.imread(abs_image_path)
        height, width = image.shape[0], image.shape[1]
        json_object['height'] = height
        json_object['width']  = width

        # read label
        bboxes = []
        labels = []
        with open(abs_label_path, 'r') as file:
            for line in file.readlines():
                splits = line.split(' ')
                cls_ = splits[0]
                if cls_ not in KITTI_NAMES:
                    continue
                labels.append(KITTI_NAMES.index(cls_) + 1) # Notice the first index is expected to be background
                bbox = [float(splits[i]) for i in range(4, 8)]
                bboxes.append(bbox)
        # bboxes = np.array(bboxes).reshape(-1, 4)
        # labels = np.array(labels).reshape(-1)
        json_object['ann'] = {
            "bboxes":bboxes, "labels":labels
        }
        output_json.append(json_object)
    json.dump(output_json, open(output_file, 'w'))

def main():
    Fire(kitti2custom)


if __name__ == '__main__':
    Fire(kitti2custom)
