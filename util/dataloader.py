import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_dataset(validation_split=0.3):
    """
    Creates a trainings and cross-validation dataset out of the original train dataset.

    :param validation_split:  The validation split as a percentage number in the range [0, 1].
    :return: train_dataset: The dataset that is used for training of the network.
    :return: valid_dataset: The dataset that is used for the cross validation in the network.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../dataset', train=True, download=True, transform=transform)
    # valid_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # shuffle dataset
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # create loader for train and validation sets
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)
    # todo: change batch size
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=valid_sampler)

    return train_loader, validation_loader


def get_test_dataset():
    """
    Get the test dataset loaded

    :return: The test dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return torch.utils.data.DataLoader(
        datasets.MNIST('../dataset', train=False, transform=transform),
        batch_size=100, shuffle=True)
