# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import sys
from copy import copy

import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.yolo import v8
from ultralytics.yolo.utils import DEFAULT_CFG, RANK
from ultralytics.yolo.utils.plotting import plot_images, plot_results


# BaseTrainer python usage
class SegmentationTrainer(v8.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'segment'
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = SegmentationModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return v8.segment.SegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png


def train(opt, cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or 'yolov8n-seg.pt'
    data = cfg.data or 'coco128-seg.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    name = cfg.name
    imgsz = cfg.imgsz
    batch = cfg.batch
    epochs = cfg.epochs
    if opt:
        data = opt.data
        imgsz = opt.imgsz
        batch = opt.batch
        epochs = opt.epochs
        model = opt.model
        name = opt.name
    args = dict(model=model, data=data, device=device, name=name, imgsz=imgsz, batch=batch, epochs=epochs)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco128-seg.yaml', help='path to data file, i.e. coco128.yaml')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--imgsz', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='number of worker threads for data loading (per RANK if DDP)')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--batch', type=int, default=256, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--name', type=str, default="", help='results saved to \'project/name\' directory')
    opt = parser.parse_args()
    print(opt)
    train(opt, use_python=False)