import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YogaKeypointLoader(data.Dataset):
    def __init__(self, root, mode, size=(512,512)):
        self.root = root
        self.classes = os.listdir(self.root)
        self.classes.sort()


        # print(self.classes)
        self.images = []
        self.labels = []
        self.points = []
        self.mode = mode
        self.size = size
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        for i in range(len(self.classes)):
            cls_path = os.path.join(self.root, self.classes[i])
            names = os.listdir(cls_path)
            for n in names:
                if not n.endswith(".txt"):
                    img_path = os.path.join(cls_path, n)
                    self.images.append(img_path)
                    self.labels.append(i)
                    self.points.append(img_path.split('.')[0]+'.txt')
                
        self.n_keypoint_coord = np.loadtxt(self.points[0]).size

        print("> Found %d images..." % (len(self.images)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.images)

    def __getitem__(self, index):
        #print(os.path.join(self.images[index]))
        image = Image.open(os.path.join(self.images[index])).convert("RGB")
        label = self.labels[index]
        point = np.loadtxt(self.points[index])

        w, h = image.size
        if self.mode == 'train':
            trans = transforms.Compose([
                transforms.Resize(self.size),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        img = trans(image)
        #print(img.shape)
        return img, label, torch.tensor(point), w, h

