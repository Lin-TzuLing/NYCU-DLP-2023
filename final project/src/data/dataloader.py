from torch.utils.data import DataLoader

from data.dataset import YogaKeypointLoader


def load_data(args, device):

    """data to dataset (img, label, keypoint coordinate, w, h)"""
    train_dataset = YogaKeypointLoader(root=args.train_path, mode="train", size=tuple(args.img_size))
    test_dataset = YogaKeypointLoader(root=args.test_path, mode="test", size=tuple(args.img_size))

    """number of object types"""
    n_classes = len(test_dataset.classes)
    n_keypoint_coord = test_dataset.n_keypoint_coord

    """dataset to dataloader"""
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize,
                             shuffle=True, num_workers=args.num_workers)
 
    
    return train_loader, test_loader, n_classes, n_keypoint_coord
