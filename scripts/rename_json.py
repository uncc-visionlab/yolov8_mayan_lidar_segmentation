import cv2
import json
import ast
import os
import numpy as np

# To run in as a class
if __name__ == "__main__":
    # os.chdir("/home.md1/jzhang72/")
    folder_name = ["train", "test", "val"]
    for fn in folder_name:
        json_path = "/home/jzhang72/PycharmProjects/lidar-segmentation/matlab/yolov7_multiscale_ALS_final/annotations/" + fn + ".json"
        new_json_path = "/home/jzhang72/PycharmProjects/lidar-segmentation/matlab/yolov7_multiscale_ALS_final/annotations/" + fn + "_new.json"
        data = json.load(open(json_path))

        # Retrieve data
        annotation_key = 'annotationList'
        img_id = 'image_id'
        cat_id = 'category_id'
        for i in range(len(data[annotation_key])):
            # Get required data
            data[annotation_key][i][cat_id] = 1  # update from 2 to 1

        newData = json.dumps(data)
        with open(json_path, 'w') as file:
            # write
            file.write(newData)

