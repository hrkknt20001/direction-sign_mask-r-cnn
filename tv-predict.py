import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
import random
import shutil

import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from Dataset import Dataset, get_transform

from model import get_model_instance_segmentation

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import utils
import transforms as T

def random_color(alpha=None):
    cr = [random.randint(0, 255) for _ in range(3)]
    if alpha is not None:
        cr.append(alpha)
    return cr        

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets, img_name in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        base_img = Image.open( os.path.join('./data/Images/',img_name[0]['filename']) ).convert('RGBA')
        draw = ImageDraw.Draw(base_img)

        for idx in range(len(outputs[0]['labels'])):
            if outputs[0]['scores'][idx] < 0.75:
                continue

            draw.line(
                    ( 
                        int(outputs[0]['boxes'][idx][0]), int(outputs[0]['boxes'][idx][1]), 
                        int(outputs[0]['boxes'][idx][2]), int(outputs[0]['boxes'][idx][1]), 
                        int(outputs[0]['boxes'][idx][2]), int(outputs[0]['boxes'][idx][3]),
                        int(outputs[0]['boxes'][idx][0]), int(outputs[0]['boxes'][idx][3]),
                        int(outputs[0]['boxes'][idx][0]), int(outputs[0]['boxes'][idx][1])
                    ), 
                    fill=tuple(random_color()), width=2
                )

        base_img.save( os.path.join('./predict', img_name[0]['filename']) )
        
        print(img_name[0]['filename'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def main():

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    

    if os.path.isdir('./predict'):
        shutil.rmtree('./predict')
    os.makedirs('./predict')

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = Dataset('./data', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        #dataset, batch_size=2, shuffle=True, num_workers=4,
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)


    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    if device.type == 'cpu':
        model.load_state_dict(torch.load("model.pth", map_location=device))
    else:
        model.load_state_dict(torch.load("model.pth", map_location="cuda:0"))
        model.to(device)

    # evaluate on the test dataset
    evaluate(model, data_loader, device=device)

    print("That's it!")
    
if __name__ == "__main__":
    main()
