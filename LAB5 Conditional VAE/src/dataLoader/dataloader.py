
from torch.utils.data import DataLoader
from dataLoader.dataset import bair_robot_pushing_dataset

def load_train_data(args):
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')


    train_loader = DataLoader(train_data,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    validate_loader = DataLoader(validate_data,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True)

    train_iterator = iter(train_loader)
    validate_iterator = iter(validate_loader)

    return train_data, train_loader, train_iterator, \
        validate_data, validate_loader, validate_iterator


def load_test_data(args):
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

    test_iterator = iter(test_loader)

    return test_data, test_loader, test_iterator