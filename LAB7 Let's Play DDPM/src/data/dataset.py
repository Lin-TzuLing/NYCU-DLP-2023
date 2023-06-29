import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


"""turn label in list into onehot"""
def label_to_onehot(object_dict, label):
    onehot_label = torch.zeros((len(label), len(object_dict)))
    for i in range(len(label)):
        for object_type in label[i]:
            onehot_label[i, object_dict[object_type]]=1
    return onehot_label


class iclevr_dataset(Dataset):
    def __init__(self, args, mode, device):
        self.mode = mode
        self.device = device
        self.root = args.dataset_path
        
        # object dictionary {key:object name, value:object id}
        self.object_dict= json.load(open(os.path.join(self.root, 'objects.json')))
        # load image and label when training
        if mode=='train':
            self.train_data = json.load(open(os.path.join(self.root, 'train.json')))
            self.img_list = list(self.train_data.keys())
            self.label_list = list(self.train_data.values())
            self.onehot_label_list = label_to_onehot(self.object_dict, self.label_list)
        # load label when testing
        elif mode=='test':
            self.test_data = json.load(open(os.path.join(self.root, 'test.json')))
            self.label_list = self.test_data
            self.onehot_label_list = label_to_onehot(self.object_dict, self.label_list)
        elif mode=='test_new':
            self.test_data = json.load(open(os.path.join(self.root, 'new_test.json')))
            self.label_list = self.test_data
            self.onehot_label_list = label_to_onehot(self.object_dict, self.label_list)

        # transform image
        self.transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop([args.img_size, args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # number of object types
        self.n_classes = len(self.object_dict)


    def __len__(self):
        if self.mode=='train':
            return len(self.train_data)
        else:
            return len(self.test_data)
    

    def __getitem__(self, index):

        if self.mode=='train':
            """read image"""
            img_root = os.path.join(self.root, 'iclevr')
            img = Image.open(os.path.join(img_root, self.img_list[index])).convert('RGB')
            img = self.transform(img)
            "condition (onehot)"
            cond =  self.onehot_label_list[index]
            return img, cond
        
        elif self.mode=='test' or self.mode=='test_new':
            """condition (onehot)"""
            cond = self.onehot_label_list[index]
            return cond



    

    
    