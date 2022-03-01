# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import datetime
import numpy as np
import torch
from Dataset import Dataset, get_transform

from model import get_model_instance_segmentation

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.utils.tensorboard import SummaryWriter

def main():
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    

    writer = SummaryWriter(log_dir=f"./logs_{datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = Dataset('./data', get_transform(train=True))
    dataset_test = Dataset('./data', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        #dataset, batch_size=2, shuffle=True, num_workers=4,
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.9)

    # let's train it for 10 epochs
    num_epochs = 100

    loss = float('inf')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        writer.add_scalar("train/lr", metric_logger.meters['lr'].median, epoch)
        writer.add_scalar("train/loss", metric_logger.meters['loss'].median, epoch)
        writer.add_scalar("train/loss_classifier", metric_logger.meters['loss_classifier'].median, epoch)
        writer.add_scalar("train/loss_box_reg", metric_logger.meters['loss_box_reg'].median, epoch)
        writer.add_scalar("train/loss_mask", metric_logger.meters['loss_mask'].median, epoch)
        writer.add_scalar("train/loss_objectness", metric_logger.meters['loss_objectness'].median, epoch)
        writer.add_scalar("train/loss_rpn_box_reg", metric_logger.meters['loss_rpn_box_reg'].median, epoch)

        if metric_logger.meters['loss'].median < loss:
            torch.save(model.state_dict(), "model_best.pth")
            loss = metric_logger.meters['loss'].median
        else:
            torch.save(model.state_dict(), "model_last.pth")

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "model.pth")
    
    print("That's it!")
    
if __name__ == "__main__":
    main()
