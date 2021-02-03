"""
    This script convert a output pickle file produced by mmdetection into label file same as KITTI dataset

    mmdetection output pickle:
        A list of arraylist, the length of the list equals the number of samples tested.
        Each arraylist is a list of numpy.ndarray, the length of this list is the number of classes
        Each array has size [num_objects, 5] -> [left, top, right, bottom, scores]

    KITTI object default format:
    ```
        $obj_type -1 -1 -10 $left $top $right $bottom -1 -1 -1 -1000 -1000 -1000 -10 $score
    ```

    Example Usage:
    ```bash
    pyhelp.mmdet2kitti --pickle_file_path=<path_to_pkl>.pkl --output_dir_path=<path_to_outputlabel_dir> --score_threshold=0.4
    ```
"""

from fire import Fire
import pickle
import os
import tqdm
def mmdet2kitti(pickle_file_path:str,
                output_dir_path:str,
                score_threshold:float=0.4,
                class_names:list=['Car', 'Pedestrian', 'Cyclist']):
    result_object = pickle.load(open(pickle_file_path, 'rb'))
    if not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    for i in tqdm.tqdm(range(len(result_object))):
        file_path = os.path.join(output_dir_path, "%06d.txt" % i)
        result_array_list = result_object[i]
        with open(file_path, 'w') as file:
            lines_to_write = []
            for class_name, results in zip(class_names, result_array_list):
                # results : [num_objects, 5]
                strong_results = results[results[:, 4] > score_threshold] #[num_objects, 5]
                for obj in strong_results:
                    lines_to_write.append( "{} -1 -1 -10 {} {} {} {} -1 -1 -1 -1000 -1000 -1000 -10 {}\n".format(
                        class_name, obj[0], obj[1], obj[2], obj[3], obj[4]
                    )
                    )
            file.writelines(lines_to_write)

def main():
    Fire(mmdet2kitti)

if __name__ == "__main__":
    Fire(mmdet2kitti)
