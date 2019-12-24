
import random
import time
import numpy as np
import torch
import math

import cv2 #changedByAlex
#Importing library to do image related operations
from PIL import Image, ImageOps


#Importing the important functionalities of Pytorch such as the dataloader, Variable, transform's 
#and optimizer related functions.

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor, ToPILImage


# Importing the dataset class for VOC12 and cityscapes
from dataset import cityscapes
from dataset import idd_lite

import sys

#Importing the Relabel, ToLabel and Colorize class from transform.py file
from transform import Relabel, ToLabel, Colorize
import matplotlib
from matplotlib import pyplot as plt


import importlib
from iouEval import iouEval, getColorEntry #importing iouEval class from the iouEval.py file
from shutil import copyfile


# ### A few global parameters ###


NUM_CHANNELS = 3 #RGB Images
NUM_CLASSES = 2 #IDD Lite has 8 labels or Level1 hierarchy of labels
USE_CUDA = torch.cuda.is_available()
IMAGE_HEIGHT = 256
OUTPUT_DIR = r'D:\Users Data\inbal.tlgip\Project\output_images'
DATA_ROOT = r'D:\Users Data\inbal.tlgip\Desktop\ChangeDetectionDataset'
BATCH_SIZE = 2
NUM_WORKERS = 10
NUM_EPOCHS = 20
ENCODER_ONLY = False
device = torch.device("cuda")
# device = torch.device("cpu")
device = 'cuda'
color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


IOUTRAIN = False
IOUVAL = True


#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=160):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, input1, target):
        # Resizing data to required size
        input =  Resize((self.height,320), Image.BILINEAR)(input) #askalex
        input1 =  Resize((self.height,320), Image.BILINEAR)(input1) #ChangedByUs
        target = Resize((self.height,320), Image.NEAREST)(target)

        if(self.augment):
            # Random horizontal flip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                input1 = input1.transpose(Image.FLIP_LEFT_RIGHT) #ChangedByUs
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            #Random translation 0-2 pixels (fill rest with padding)
            transX = random.randint(0, 2)
            transY = random.randint(0, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            input1 = ImageOps.expand(input1, border=(transX,transY,0,0), fill=0) #ChangedByUs
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 7  #askalex- change 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            input1 = input1.crop((0, 0, input1.size[0]-transX, input1.size[1]-transY)) #ChangedByUs
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        input1 = ToTensor()(input1) #ChangedByUs

        target = ToLabel()(target)

        #target = Relabel(255, 7)(target)
        return input, input1, target #ChangedByUs


# ### Loading Data
#
# We'll follow pytorch recommended semantics, and use a dataloader to load the data.



def main():
    best_acc = 0

    co_transform = MyCoTransform(ENCODER_ONLY, augment=True, height=IMAGE_HEIGHT)
    co_transform_val = MyCoTransform(ENCODER_ONLY, augment=False, height=IMAGE_HEIGHT)

    #train data
    dataset_train = idd_lite(DATA_ROOT, co_transform, 'train')
    print("length of training set: ",len(dataset_train))
    #test data
    dataset_val = idd_lite(DATA_ROOT, co_transform_val, 'val')
    print("length of validation set: ",len(dataset_val))


    # NOTE: PLEASE DON'T CHANGE batch_size and num_workers here. We have limited resources.
    loader_train = DataLoader(dataset_train, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    dataiter = iter(loader_val)
    (first_val_image_A, first_val_image_B, first_val_image_labels) = dataiter.next()
    first_val_image_A = first_val_image_A.to(device)
    first_val_image_B = first_val_image_B.to(device)
    # ## Cross Entropy  Loss ##
    # Negative Log Loss   |Plot of -log(x) vs x
    # - | -
    # ![alt](img/nll.png) | ![alt](img/nll-log.png)
    #
    # The negative log-likelihood becomes unhappy at smaller values, where it can reach infinite unhappiness (that’s too sad), and becomes less unhappy at larger values. Because we are summing the loss function to all the correct classes, what’s actually happening is that whenever the network assigns high confidence at the correct class, the unhappiness is low, but when the network assigns low confidence at the correct class, the unhappiness is high.

    # In[12]:


    criterion = torch.nn.CrossEntropyLoss()



    #get some random training images
    print("length of training couples: ",len(loader_train))
    print(len(loader_val))
    dataiter = iter(loader_train)
    (images, images1, labels) = dataiter.next() #ChangedByUs
    # for step, (images, labels) in enumerate(loader_train):
    # plt.figure()
    # plt.imshow(ToPILImage()(images[0].cpu()))
    # plt.figure()
    # plt.imshow(ToPILImage()(Colorize()(labels[0].cpu())))
    # break


    # ## Model ##


    model_file = importlib.import_module('erfnet')
    model = model_file.Net(NUM_CLASSES).to(device)



    # ### Optimizer ###


    # We use adam optimizer. It can be replaced with SGD and other optimizers
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    start_epoch = 1


    print("device used: ",device)


    # ### Training Procedure ###
    softmax = torch.nn.Softmax(dim=1)
    import os
    steps_loss = 50
    my_start_time = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []

        doIouTrain = IOUTRAIN
        doIouVal =  IOUVAL

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        model.train()
        for step, (images, images1, labels) in enumerate(loader_train): #ChangedByUs
            start_time = time.time()
            # inputs = [images.to(device), images1.to(device)] #ChangedByUs
            inputs = images.to(device)
            inputs1 = images1.to(device) #ChangedByUs
            targets = labels.to(device)
            targets_orig = targets.clone()
            targets[targets_orig >= 128] = 1  # ChangedByUs
            targets[targets_orig < 128] = 0  # ChangedByUs
            #for x_u in targets.unique():
            #    print(int(x_u), ' appears ', int(torch.stack([(targets==x_u).sum()])), ' times.\n')
            outputs = model([inputs, inputs1], only_encode=ENCODER_ONLY)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            # print statistics
            if steps_loss > 0 and step % steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {average:',average,'} (epoch: {',epoch,'}, step: {', step, '})', "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / BATCH_SIZE))


            if step % 500 == 0:
                outputs_val = model([first_val_image_A.cuda(), first_val_image_B.cuda()], only_encode=ENCODER_ONLY)
                outputs_val = softmax(outputs_val)
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'epoch'+str(epoch)+'_iter' + str(step) + '_A.tiff'),
                            np.rollaxis((first_val_image_A[0, :, :, :].squeeze().cpu().numpy() * 255).astype('uint8'), 0, 3))
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'epoch'+str(epoch)+'_iter' + str(step) + '_B.tiff'),
                            np.rollaxis((first_val_image_B[0, :, :, :].squeeze().cpu().numpy() * 255).astype('uint8'), 0, 3))
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'epoch'+str(epoch)+'_iter' + str(step) + '_label.tiff'),
                            (first_val_image_labels[0, :, :, :].squeeze().cpu().numpy()).astype('uint8'))
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'epoch'+str(epoch)+'_iter' + str(step) + '_output.tiff'),
                            (((outputs_val[0, 1, :, :] > 0.5) * 255).squeeze().cpu().numpy()).astype('uint8'))


        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")

        #save one image per epoch
        if USE_CUDA:
            first_val_image_A = first_val_image_A.to(device)
            first_val_image_B = first_val_image_B.to(device)  # ChangedByUs
            first_val_image_labels = first_val_image_labels.to(device)

        inputs = first_val_image_A.to(device)
        inputs1 = first_val_image_B.to(device)  # ChangedByUs

        with torch.no_grad():
            outputs = model([inputs, inputs1], only_encode=ENCODER_ONLY)  # ChangedByUs

        label = outputs[0].max(0)[1].byte().cpu().data

        label_color = Colorize()(label.unsqueeze(0))

        label_save = ToPILImage()(label_color)

        plt.figure()
        plt.imshow(label_save)
        plt.savefig(r'D:\Users Data\inbal.tlgip\Project\output_images\epoch' + str(epoch) + '.jpg')

    my_end_time = time.time()
    print(my_end_time - my_start_time)


    print('loss: {average:',average,'} (epoch: {',epoch,'}, step: {',step,'})', "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / BATCH_SIZE))


    # # ### Validation ###
    # #Validate on val images after each epoch of training
    # print("----- VALIDATING - EPOCH", epoch, "-----")
    # model.eval()
    # epoch_loss_val = []
    # time_val = []
    #
    # if (doIouVal):
    #     iouEvalVal = iouEval(NUM_CLASSES)
    #
    # for step, (images, labels) in enumerate(loader_val):
    #     start_time = time.time()
    #
    #     inputs = images.to(device)
    #     targets = labels.to(device)
    #
    #     with torch.no_grad():
    #         outputs = model(inputs, only_encode=ENCODER_ONLY)
    #         #outputs = model(inputs)
    #     loss = criterion(outputs, targets[:, 0])
    #     epoch_loss_val.append(loss.item())
    #     time_val.append(time.time() - start_time)
    #
    #
    #     #Add batch to calculate TP, FP and FN for iou estimation
    #     if (doIouVal):
    #         #start_time_iou = time.time()
    #         iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
    #         #print ("Time to add confusion matrix: ", time.time() - start_time_iou)
    #
    #     if steps_loss > 0 and step % steps_loss == 0:
    #         average = sum(epoch_loss_val) / len(epoch_loss_val)
    #         print('VAL loss: {average:',average,'} (epoch: {',epoch,'}, step: {',step,'})',
    #                 "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / BATCH_SIZE))
    #
    #
    # average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
    #
    # iouVal = 0
    # if (doIouVal):
    #
    #     iouVal, iou_classes = iouEvalVal.getIoU()
    #     print(iou_classes)
    #     iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    #     print ("EPOCH IoU on VAL set: ", iouStr, "%")

    #
    #  ### Visualizing the Output###
    torch.save(model.state_dict(),r'C:\Users\inbal.tlgip\modelsave.pt')
    # Qualitative Analysis





if __name__ == '__main__':
    #freeze_support()
    main()