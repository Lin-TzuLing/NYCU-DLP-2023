import pandas as pd
from torch.utils import data
import numpy as np

import os
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


def create_dataset(args, device):
    """turn data into dataset"""

    """data to dataset"""
    train_dataset = RetinopathyLoader(root=os.path.join(args.dataset_path,"new_train"),
                                      mode="train", data_path=args.data_path)
    test_dataset = RetinopathyLoader(root=os.path.join(args.dataset_path,"new_test"),
                                      mode="test", data_path=args.data_path)

    """number of classes in this task"""
    num_classes = train_dataset.num_classes

    "batch slicing, shuffle only training data, on device"
    train_dataset = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=2)
                                    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_dataset = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=2)
                                   # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    return train_dataset, test_dataset, num_classes




def getData(mode, path):
    """list of img name and its corresponding label"""
    if mode == 'train':
        img = pd.read_csv(os.path.join(path,'train_img.csv'))
        label = pd.read_csv(os.path.join(path,'train_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(os.path.join(path,'test_img.csv'))
        label = pd.read_csv(os.path.join(path,'test_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, data_path):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)
            data_path : Root path of the train/test_img.csv and train/test_label.csv

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode, data_path)
        self.mode = mode
        # number of classes in dataset
        self.num_classes = len(set(self.label))
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # step 1
        path = os.path.join(self.root, self.img_name[index]+'.jpeg')
        # step 2
        label = float(self.label[index])
        # step 3
        img = Image.open(path)

        transform = transforms.Compose(
            [
                transforms.Resize(512),
                # crop to same size
                transforms.CenterCrop([512, 512]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # ToTensor() convert pixel value to [0,1] and transpose to [C,H,W]
                transforms.ToTensor(),
                transforms.Normalize((0.3749, 0.2602, 0.1857), (0.2526, 0.1780, 0.1291)),
            ]
        )
        img = transform(img)
        return img, label
