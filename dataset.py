import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import ToTensor, ToPILImage

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelCatIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, basename+extension)

def image_path_city(root, name):
    return os.path.join(root, name)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes_mine(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        #print (self.images_root)
        
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        #print('Imported PDB')
        #import pdb;pdb.set_trace()
        self.filenamesGt.sort()
        #print(self.filenamesGt)
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        #print(filename[61:-16])
        #print(filenameGt[56:-25])
        #if filename[61:-16] != filenameGt[56:-25]:
            #print("nooo..")
            #print(filename[61:-16])
            #print(filenameGt[56:-25])
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        
        oldimage = image
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        #print(self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        
        #print(self.filenames)

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        #print(self.labels_root)
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        #self.filenamesGt = [os.path.join(dp, f) for f in os.listdir(self.labels_root) if is_label(f)]
        
        #print('Imported PDB')
        #import pdb;pdb.set_trace()
        #print(self.filenamesGt)
        self.filenamesGt.sort()
        
        #print(self.filenamesGt)

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        #print(len(self.filenames))
        filename = self.filenames[index]
        #print('Index is:  ', index)
        filenameGt = self.filenamesGt[index]
        #print(filename[61:-16])
        #print(filenameGt[56:-25])
        '''if filename[61:-16] != filenameGt[56:-25]:
            print("nooo..")
            print(filename[61:-16])
            print(filenameGt[56:-25])'''
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        
        oldimage = image
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    
    
class idd_lite(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'images/')
        self.labels_root = os.path.join(root, 'labels/')
        self.images_root += subset
        self.labels_root += subset
        self.filenames1 = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root + '/A')) for f in fn if is_image(f)]
        self.filenames1.sort()

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root + '/B')) for f in
                          fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
        self.filenamesGt.sort()
        
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        
        filename = self.filenames[index]
        filename1 = self.filenames1[index] #ChangedByUs
        filenameGt = self.filenamesGt[index]
        
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.images_root, filename1), 'rb') as f: #ChangedByUs
            image1 = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        
        oldimage = image
        if self.co_transform is not None:
            image, image1, label = self.co_transform(image, image1, label) #ChangedByUs

        return image, image1, label #ChangedByUs

    def __len__(self):
        return len(self.filenames)
    

