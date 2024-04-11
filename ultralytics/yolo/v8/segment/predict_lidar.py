# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

import argparse
import sys

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
from skimage import morphology

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.plotting import Colors

colors = Colors()


class LidarSegmentationPredictor:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = None
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.batch = None
        self.results = None
        self.transforms = None

    def setup_model(self, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model = self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            if len(im.shape) == 3:
                im = np.expand_dims(im, axis=0)  # HWC to BHWC
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, input_img_path, scale_factor, size_filter):
        # preds (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
        # containing the predicted boxes, classes, and masks. The tensor should be in the format
        # output by a model, such as YOLO.
        p = ops.non_max_suppression(preds,
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        # p: (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
        #             shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
        #             (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        min_wh = size_filter
        # max_wh = 200 * scale_factor
        for i, pred in enumerate(p):  # i = image index in a batch, pred = image inference (a tensor of shape (num_boxes, 6 + num_masks))
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            img_path = input_img_path if input_img_path else None
            if len(pred) == 0:  # empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:  # use high-resolution segmentation masks. See the configuration YAML file.
                if not isinstance(orig_img, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # return masks with dimensions [n, h, w]
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # return a binary mask of shape [n, h, w]
                if not isinstance(orig_img, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # filter out predictions by the size of the bounding boxes
            good_idx = np.ones(len(pred))
            for idx in range(len(pred)):
                pred_cls = pred[idx, 5]
                if pred_cls == 1.0:  # for platforms only
                    # width-height from bounding box (slightly larger compared to the mask width-height)
                    # width = pred[idx, 2] - pred[idx, 0]
                    # height = pred[idx, 3] - pred[idx, 1]
                    # print("bbox height", height)
                    # print("bbox width", width)
                    """
                    ------------ DEBUG masks inside bbox ----------------
                    print("bbox top-left corner", (pred[idx, 1], pred[idx, 0]))
                    print("bbox size", len(pred))
                    print("masks size", len(masks))
                    print("mask center data", masks[idx, :, :].cpu().numpy()[int(pred[idx, 1] + 0.5*height), int(pred[idx, 0] + 0.5*width)])
                    cv2.imshow("mask[0]", masks[0, :, :].cpu().numpy())
                    cv2.waitKey(0)
                    """
                    # width-height from mask (more refined, see https://discuss.pytorch.org/t/extracting-bounding-box-coordinates-from-mask/61179/7)
                    c_idx = np.where(np.any(masks[idx, :, :].cpu().numpy(), 0))[0]
                    r_idx = np.where(np.any(masks[idx, :, :].cpu().numpy(), 1))[0]
                    if not c_idx.shape[0]:  # found some cases where a bounding box exists while no mask is created within it
                        good_idx[idx] = 0
                        continue
                    cmin, cmax = c_idx[[0, -1]]
                    rmin, rmax = r_idx[[0, -1]]
                    width = cmax - cmin
                    height = rmax - rmin
                    # print("mask height", height)
                    # print("mask width", width)
                    if height < min_wh or width < min_wh:
                        good_idx[idx] = 0
            # print(pred.shape)
            pred = pred[good_idx.astype(bool)]
            masks = masks[good_idx.astype(bool)]
            # print("new shape = ", pred.shape)
            if len(pred) == 0:  # if the prediction tensor is empty after removing all the small bounding boxes
                continue

            # boxes=pred[:, :6]: boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            # with shape (num_boxes, 6). The last two columns should contain confidence and class values.
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def predict(self, orig_img, img_path, scale_factor, size_filter):
        im = self.preprocess(orig_img)

        # Inference
        preds = self.model(im, augment=self.args.augment, visualize=False)

        # Postprocess
        results = self.postprocess(preds, im, orig_img, img_path, scale_factor, size_filter)
        return results


# https://github.com/ultralytics/ultralytics/issues/4011
# https://github.com/ultralytics/yolov5/issues/3607
def infer(model, imgsz, classes, conf):
    if classes is None or len(classes) > 1:
        sys.exit("Exit: This code currently only supports to generate mask image for different classes separately.")

    home_folder = '/home/jzhang72/PycharmProjects/lidar-segmentation/'
    results_folder = '/home/jzhang72/PycharmProjects/ultralytics/inference_results/'
    gis_output_image_filenames = ['yolo_KOM_image_segmented.png',
                                  'yolo_MLS_image_segmented.png',
                                  'yolo_SAY_image_segmented.png',
                                  'yolo_HNT_image_segmented.png']

    gis_output_mask_filenames = ['yolo_KOM_segmented_mask.png',
                                 'yolo_MLS_segmented_mask.png',
                                 'yolo_SAY_segmented_mask.png',
                                 'yolo_HNT_segmented_mask.png']

    # ALS 3-band data
    gis_data_path = ['data/ALS_data_(filtered_polygons)/KOM/', 'data/ALS_data_(filtered_polygons)/MLS/',
                     'data/ALS_data_(filtered_polygons)/Sayil_clipped/', 'data/ALS_data_(filtered_polygons)/HNT/']

    gis_input_filenames_hs = ['kom_dsm_lidar_hs.png',
                              'MLS_DEM_hs.png',
                              'Sayil_clipped_DEM_hs.png',
                              'HNT_4-5km2_HS.png']  # for visualization purpose

    gis_input_filenames_mat = ['KOM_ALS_data.mat',
                               'MLS_ALS_data.mat',
                               'Sayil_ALS_data.mat',
                               'HNT_4-5km2_ALS_data.mat']

    output_folder = results_folder
    os.makedirs(output_folder, exist_ok=True)

    """Runs YOLO model inference on input image(s)."""
    model = model or 'yolov8n.pt'
    imgsz = imgsz
    args = dict(model=model, imgsz=imgsz, classes=classes, conf=conf)

    """Model"""
    predictor = LidarSegmentationPredictor(overrides=args)
    predictor.setup_model()

    """Data"""
    IMAGE_SIZE = imgsz
    print(IMAGE_SIZE)
    rescale = False  # rescale the input image
    scale_factor = [1.0]  # scale needs to be a list
    if rescale:
        scale_factor = scale_factor + [2.0]  # append more factors
    post_processing_size_filter = 15.0  # this value will be multiplied by the scaling factor

    for DATASET_INDEX in [3]:  # range(len(gis_input_filenames_mat)):
        print("Inference on " + gis_input_filenames_mat[DATASET_INDEX].split('.')[0])
        img_filename_mat = home_folder + gis_data_path[DATASET_INDEX] + gis_input_filenames_mat[DATASET_INDEX]
        if DATASET_INDEX == "":  # if the dataset MAT file is too big, then use h5py to read, for example, Sayil
            with h5py.File(img_filename_mat, 'r') as f:
                raw_image_data = np.array(f['geotiff_data']).transpose()
        else:
            mat_data = sio.loadmat(img_filename_mat, squeeze_me=True)
            raw_image_data = mat_data['geotiff_data']

        output_filename = output_folder + gis_output_image_filenames[DATASET_INDEX]
        output_mask_filename = output_folder + gis_output_mask_filenames[DATASET_INDEX]
        raw_img_filename_hs = home_folder + gis_data_path[DATASET_INDEX] + gis_input_filenames_hs[DATASET_INDEX]
        raw_image_data_hs = cv2.imread(raw_img_filename_hs)

        h = raw_image_data.shape[0]
        w = raw_image_data.shape[1]
        # a final predicted mask that is stacked from maks images at different scale
        multi_scale_predicted_mask = np.zeros((raw_image_data.shape[0], raw_image_data.shape[1], 3), dtype=np.float32)

        xy_pixel_skip = (80, 80)
        xy_pixel_margin = np.array([np.round((IMAGE_SIZE + 1) / 2), np.round((IMAGE_SIZE + 1) / 2)], dtype=np.int32)

        # rescale image data and hillshade data for inference
        for sf in scale_factor:
            print("Scaling factor = ", sf)
            if sf != 1.0:
                image_data = cv2.resize(raw_image_data, (int(sf * w), int(sf * h)), interpolation=cv2.INTER_CUBIC)
                image_data_hs = cv2.resize(raw_image_data_hs, (int(sf * w), int(sf * h)), interpolation=cv2.INTER_CUBIC)
            else:
                image_data = raw_image_data
                image_data_hs = raw_image_data_hs

            [rows, cols] = image_data.shape[0:2]

            x_vals = range(xy_pixel_margin[0], cols - xy_pixel_margin[0], xy_pixel_skip[0])
            y_vals = range(xy_pixel_margin[1], rows - xy_pixel_margin[1], xy_pixel_skip[1])

            label_image_predicted = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.float32)
            empty_image = np.zeros((imgsz, imgsz, 3), dtype=np.int8)

            for y in y_vals:
                for x in x_vals:
                    print("(x,y) = " + "(" + str(x) + ", " + str(y) + ")")
                    test_image_hs = image_data_hs[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                                    (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])]
                    test_image = image_data[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                                 (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])]  # float data
                    input_test_image = np.zeros(test_image.shape, dtype=np.uint8)
                    if len(test_image.shape) == 2:
                        # convert floating point data to regular uint255 RGB image (be consistent with the training data)
                        image_range = (np.max(test_image) - np.min(test_image))
                        if image_range == 0:
                            image_range = 1
                        test_image = (test_image - np.min(test_image)) / image_range
                        test_image = np.float32(test_image)
                        test_image = cv2.normalize(test_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        input_test_image = np.stack((test_image,) * 3, axis=-1)
                    elif len(test_image.shape) == 3:
                        for ch in range(test_image.shape[2]):
                            test_image_data_channel = test_image[:, :, ch]
                            image_range = (np.max(test_image_data_channel) - np.min(test_image_data_channel))
                            if image_range == 0:
                                image_range = 1
                            test_image_data_channel = (test_image_data_channel - np.min(test_image_data_channel)) / image_range
                            test_image_data_channel = np.float32(test_image_data_channel)
                            test_image_data_channel_norm = cv2.normalize(test_image_data_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                                         dtype=cv2.CV_8UC1)
                            input_test_image[:, :, ch] = test_image_data_channel_norm
                        # convert RGB to BGR so that the image input data is consistent with the training images loaded using cv2.imread()
                        tmp = input_test_image[:, :, 2].copy()
                        input_test_image[:, :, 2] = input_test_image[:, :, 0]
                        input_test_image[:, :, 0] = tmp
                    """predict"""
                    pred_results = predictor.predict(input_test_image, img_filename_mat, scale_factor=sf,
                                                     size_filter=post_processing_size_filter * sf)
                    if len(pred_results):
                        pred_plotted = pred_results[0].plot(img=empty_image, probs=False, boxes=False, labels=False,
                                                            kpt_line=False)  # plot results on an empty image or the hs image
                        # cv2.imshow("result", pred_plotted)
                        # cv2.waitKey(0)
                        label_image_predicted[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                        (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0]), :] += pred_plotted

                        # classification_count_image[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                        # (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])] += 1
            if sf != 1.0:
                label_image_predicted = cv2.resize(label_image_predicted, (w, h), interpolation=cv2.INTER_CUBIC)
            # stack predicted maks from different scales
            multi_scale_predicted_mask += label_image_predicted

        # binarize the image to generate the mask
        mask = 1.0 * (multi_scale_predicted_mask > 0)
        # cv2.imshow("mask", mask)      # binary mask image
        # cv2.waitKey(0)
        # post-process the mask to remove small objects
        min_thickness = int(post_processing_size_filter)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_thickness, min_thickness))
        mask = cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_OPEN, se)
        mask = morphology.remove_small_objects(np.asarray(mask, dtype=bool), min_thickness * min_thickness)

        annu_color = colors(0, True)
        platform_color = colors(1, True)
        label_color = annu_color if classes == [0] else platform_color
        output_mask = mask * 255
        output_mask = np.asarray(output_mask, dtype=np.uint8)

        cv2.imwrite(output_mask_filename, output_mask)
        maskgray = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(maskgray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]
        print("A total of " + str(len(cnt)) + " targets are found.")

        label_image_predicted = np.array(mask, dtype=np.uint8)
        label_image_predicted[(label_image_predicted == 1).all(-1)] = label_color
        output_label_predicted = cv2.addWeighted(raw_image_data_hs, 0.8, label_image_predicted, 0.4, 0)

        cv2.imwrite(output_filename, output_label_predicted)
        print("Prediction completed.")

    print("Dataset", DATASET_INDEX, "inference completed. Results saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    opt = parser.parse_args()
    print(opt)

    MODEL_PATH = opt.model
    IMAGE_SIZE = opt.imgsz
    CLASSES = opt.classes
    CONF_THRESHOLD = opt.conf_thres
    infer(MODEL_PATH, IMAGE_SIZE, CLASSES, CONF_THRESHOLD)
