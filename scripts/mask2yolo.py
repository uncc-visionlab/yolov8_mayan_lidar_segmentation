import glob
import os

import numpy as np
import cv2
from imantics import Polygons, Mask


def mask_to_polygon(mask_path, class_id):
    mask = 1.0 - cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    mask = np.asarray(mask).astype(np.int8)
    polygons = Mask(mask).polygons()
    # bbox = polygons.points  # polygon in point format [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]]]
    contours = polygons.segmentation  # polygon in segmentation format [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    labels = []
    for i in range(len(contours)):
        # norm_polygon = np.asarray(class_id).reshape(1, 1)
        obj_contours = contours[i]
        if len(obj_contours) > 5:  # at least 3 points (required by YOLO segmentation)
            x_coords = obj_contours[::2]  # odd indices
            y_coords = obj_contours[1::2]  # even indices
            obj_contours[::2] = np.round(np.asarray(x_coords) / mask.shape[1], 3)
            obj_contours[1::2] = np.round(np.asarray(y_coords) / mask.shape[0], 3)
            # norm_polygon = np.append(x_coords, y_coords)
            # norm_polygon = list(norm_polygon)
            obj_contours.insert(0, class_id)
            labels.append(obj_contours)
    # print(contours)
    # print(labels)
    return labels


def write_image_and_mask(src_path, tgt_path):
    mask_name = ['aguada', 'platform']  #['aguada', 'building', 'platform']
    image_src_dir = src_path + "/Chactun_ML_ready_lidar/lidar/"
    # create dataset list .txt file
    f_train = open(tgt_path + "/train.txt", 'wt')
    f_val = open(tgt_path + "/val.txt", 'wt')
    f_test = open(tgt_path + "/test.txt", 'wt')
    # sort the list so the content of dataset splits are the same every run so the files can be overwritten every time
    image_list = sorted(glob.glob(image_src_dir + '*.tif'))
    cnt = 0
    for filepath in image_list:
        image_filename = filepath.rsplit("/", 1)[1]
        basename = image_filename.rsplit(".", 1)[0]
        image_name = basename.rsplit("_", 1)[0]
        cnt = cnt + 1
        if cnt < 0.8 * len(image_list):
            dataset = 'train'
            f_train.write("./images/train/" + image_name + ".png" + "\n")
        else:
            dataset = 'val'
            f_val.write("./images/val/" + image_name + ".png" + "\n")
        mask_tgt_path = tgt_path + "/labels/" + dataset + "/"
        os.makedirs(mask_tgt_path, exist_ok=True)
        image_tgt_path = tgt_path + "/images/" + dataset + "/"
        os.makedirs(image_tgt_path, exist_ok=True)
        image_tgt_path = image_tgt_path + image_name + ".png"

        # create image dataset
        img = cv2.imread(filepath)
        cv2.imwrite(image_tgt_path, img)

        # create segmentation labels
        segmentation = []
        for cls in range(len(mask_name)):
            mask_src_path = src_path + "/Chactun_ML_ready_masks/masks/" + image_name + "_mask_" + mask_name[cls] + ".tif"
            cls_mask = mask_to_polygon(mask_src_path, cls)
            segmentation.append(cls_mask)  # list of array

        mask_tgt_path = mask_tgt_path + image_name + ".txt"
        f = open(mask_tgt_path, 'wt')
        try:
            for i in range(len(segmentation)):  # class
                if segmentation[i] is not None:  # class ID + polygon
                    for item in segmentation[i]:
                        f.write(" ".join(str(elem) for elem in item) + '\n')
        finally:
            f.close()

    f_train.close()
    f_val.close()
    f_test.close()


src_path = "/home/jzhang72/Downloads/Mayan_data"
tgt_path = "/home/jzhang72/PycharmProjects/ultralytics/Mayan_data_YOLO"
os.makedirs(tgt_path, exist_ok=True)
write_image_and_mask(src_path, tgt_path)
