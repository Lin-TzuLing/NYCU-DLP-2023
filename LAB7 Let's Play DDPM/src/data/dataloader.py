from torch.utils.data import DataLoader
from data.dataset import iclevr_dataset


def load_train_data(args, device):

    """data to dataset"""
    train_dataset = iclevr_dataset(args=args, mode='train', device=device)
    test_dataset = iclevr_dataset(args=args, mode='test', device=device)
    test_new_dataset = iclevr_dataset(args=args, mode='test_new', device=device)
    print('# of training data:{}'.format(len(train_dataset)))
    print('# of testing data:{}, {}'.format(len(test_dataset), len(test_new_dataset)))

    """number of object types"""
    n_classes = train_dataset.n_classes

    """dataset to dataloader"""
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    test_new_loader = DataLoader(test_new_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    
    return train_loader, test_loader, test_new_loader, n_classes


def load_test_data(args, device):

    """data to dataset"""
    test_dataset = iclevr_dataset(args=args, mode='test', device=device)
    test_new_dataset = iclevr_dataset(args=args, mode='test_new', device=device)
    print('# of testing data:{}, {}'.format(len(test_dataset), len(test_new_dataset)))

    """number of object types"""
    n_classes = test_dataset.n_classes

    """dataset to dataloader"""
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    test_new_loader = DataLoader(test_new_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    
    return test_loader, test_new_loader, n_classes