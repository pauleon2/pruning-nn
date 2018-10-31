import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pruninig_nn.network import NeuralNetwork
from pruninig_nn.pruning import PruneNeuralNetStrategy, weight_based_pruning
from pruninig_nn.util import train, test

# constant variables
hyper_params = {
    'pruning_percentage': 0.5,  # percentage of weights pruned
    'batch_size': 64,
    'test_batch_size': 100,
    'num_epochs': 10,
    'num_retrain_epochs': 5,
    'learning_rate': 0.001,
    'momentum': 0,
    'hidden_units': 100
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load trainings dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=hyper_params['batch_size'],
                                           shuffle=True)

# load test dataset
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=hyper_params['test_batch_size'],
                                          shuffle=True)

# create neural net and train (input is image, output is number between 0 and 9.
model = NeuralNetwork(28 * 28, hyper_params['hidden_units'], 10)

# Criterion and optimizer
# might actually use MSE Error
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=hyper_params['learning_rate'],
                            momentum=hyper_params['momentum'])

# train and test the network
for epoch in range(hyper_params['num_epochs']):
    train(train_loader, model, optimizer, criterion, epoch, hyper_params['num_epochs'])
    test(test_loader, model)

# prune using strategy
strategy = PruneNeuralNetStrategy(weight_based_pruning)
strategy.prune(model, hyper_params['pruning_percentage'])

# Reevaluate the network performance
test(test_loader, model)

# Retrain and reevaluate
for epoch in range(hyper_params['num_retrain_epochs']):
    train(train_loader, model, optimizer, criterion, epoch, hyper_params['num_retrain_epochs'])
    test(test_loader, model)
