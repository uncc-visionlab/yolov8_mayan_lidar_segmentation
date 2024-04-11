import cv2
import json
import ast
import os
import numpy as np


class ConvertCOCOToYOLO:
    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:
        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }

    """

    def __init__(self, img_folder, json_path):
        self.img_folder = img_folder
        self.json_path = json_path

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('error!', img_path)
            exit()

    def sorting(self, l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x, y, width, height) COCO format to
        (x_c, y_c, width, height) normalized YOLO format.
        """

        # print(img_path + "\n")
        size = self.get_img_shape(img_path)
        xmax, xmin = self.sorting(x1, x2)
        ymax, ymin = self.sorting(y1, y2)
        dw = 1. / size[1]
        dh = 1. / size[0]
        x_c = (xmin + xmax) / 2.0
        y_c = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        x_c = x_c * dw
        w = w * dw
        y_c = y_c * dh
        h = h * dh
        return x_c, y_c, w, h

    def convert(self, task, annotation_key='annotationList', img_id='image_id', cat_id='category_id', bbox='bb', seg='segmentation'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        # Create image list file
        img_list_filename = self.img_folder.rsplit("/", 2)[0] + "/" + self.img_folder.rsplit("/", 1)[1] + ".txt"
        img_list_file = open(img_list_filename, "w")
        for i in range(len(data['imageList'])):
            image_path = './images/' + self.img_folder.rsplit("/", 1)[1] + "/" + data['imageList'][i]['filename']
            img_list_file = open(img_list_filename, "a")
            img_list_file.write(image_path)
            img_list_file.write("\n")
            img_list_file.close()

        check_set = set()

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = f'{data[annotation_key][i][img_id]}'
            image_name = 'img_' + image_id.zfill(5)
            category_id = f'{data[annotation_key][i][cat_id] - 1}'  # as YOLO class starts from 0 while matlab labeling starts from 1
            bbox_str = f'{data[annotation_key][i][bbox]}'
            bbox_list = ast.literal_eval(bbox_str)
            seg_str = f'{data[annotation_key][i][seg]}'
            seg_list = ast.literal_eval(seg_str)

            print(self.img_folder + ": " + image_id)

            # if category_id == '0':  # skip annular structures
            #     continue
            # elif category_id == '1':  # platforms
            #     category_id = '0'  # assign label "0" to platforms since annular structures are skipped

            # Retrieve image.
            if self.img_folder is None:
                image_path = f'{image_name}.png'
            else:
                image_path = f'{self.img_folder}/{image_name}.png'

            # Prepare for export
            label_folder = self.img_folder.rsplit("/", 2)[0] + "/labels/" + self.img_folder.rsplit("/", 1)[1]
            os.makedirs(label_folder, exist_ok=True)
            filename = label_folder + "/" + f'{image_name}.txt'

            # Convert the data
            if task == "detection":
                coco_bbox = [bbox_list["x"], bbox_list["y"], bbox_list["x"] + bbox_list["width"], bbox_list["y"] + bbox_list["height"]]
                yolo_bbox = self.convert_labels(image_path, coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3])
                content = f"{category_id} {round(yolo_bbox[0], 4)} {round(yolo_bbox[1], 4)} {round(yolo_bbox[2], 4)} {round(yolo_bbox[3], 4)}"
            elif task == "segmentation":
                size = self.get_img_shape(image_path)
                dw = 1. / size[1]
                dh = 1. / size[0]
                yolo_seg = (np.array(seg_list).reshape(-1, 2) * np.array([dw, dh])).reshape(-1).tolist()
                yolo_seg = ' '.join([str(round(elem, 4)) for i, elem in enumerate(yolo_seg)])
                content = f"{category_id} {yolo_seg}"

            # Export
            if image_name in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filename, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_name not in check_set:
                check_set.add(image_name)
                # Write files
                file = open(filename, "w")
                file.write(content)
                file.close()


# To run in as a class
if __name__ == "__main__":
    # os.chdir("/home.md1/jzhang72/")
    task = input("Please enter the task (supported tasks are \"detection\" and \"segmentation\"): ")
    while task not in ["detection", "segmentation"]:
        task = input("Unsupported task. Please enter either \"detection\" or \"segmentation\": ")
    folder_name = ["train", "test", "val"]
    for fn in folder_name:
        img_folder = "/home/jzhang72/PycharmProjects/lidar-segmentation/matlab/platform_scale_2.0_HS_final_v3/images/" + fn
        json_path = "/home/jzhang72/PycharmProjects/lidar-segmentation/matlab/platform_scale_2.0_HS_final_v3/annotations/" + fn + ".json"
        ConvertCOCOToYOLO(img_folder, json_path).convert(task=task)
