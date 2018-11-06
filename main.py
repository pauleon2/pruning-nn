import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pruninig_nn.network import NeuralNetwork
from pruninig_nn.pruning import PruneNeuralNetStrategy
from pruninig_nn.util import train, test

# constant variables
hyper_params = {
    'pruning_percentage': 0.2,  # percentage of weights pruned
    'batch_size': 64,
    'test_batch_size': 100,
    'num_epochs': 20,
    'num_retrain_epochs': 10,
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

# save the current model
torch.save(model, './model/model.pt')
print('Saved pretrained model')

sns.set()
current_pruning_rate = hyper_params['pruning_percentage']

s = pd.DataFrame(columns=['epoch', 'accuracy', 'pruning_perc'])
while current_pruning_rate < 0.9:
    # loading the model
    loaded_model = torch.load('./model/model.pt')
    loaded_model.train()

    criterion_loaded = nn.NLLLoss()
    optimizer_loaded = torch.optim.SGD(loaded_model.parameters(),
                                       lr=hyper_params['learning_rate'],
                                       momentum=hyper_params['momentum'])

    # prune using strategy
    strategy = PruneNeuralNetStrategy()
    strategy.prune(loaded_model, current_pruning_rate)

    accuracy = np.zeros(hyper_params['num_retrain_epochs'] + 1)
    # Reevaluate the network performance
    accuracy[0] = test(test_loader, loaded_model)

    # Retrain and reevaluate
    for epoch in range(hyper_params['num_retrain_epochs']):
        train(train_loader, loaded_model, optimizer_loaded, criterion_loaded, epoch, hyper_params['num_retrain_epochs'])
        accuracy[epoch + 1] = test(test_loader, loaded_model)

    tmp = pd.DataFrame({'epoch': range(0, hyper_params['num_retrain_epochs'] + 1),
                        'accuracy': accuracy,
                        'pruning_perc': np.full(hyper_params['num_retrain_epochs'] + 1, current_pruning_rate)})
    s = s.append(tmp, ignore_index=True)

    # update current pruning rate
    current_pruning_rate = current_pruning_rate + hyper_params['pruning_percentage']

plot = sns.relplot(x='epoch', y='accuracy', hue="pruning_perc",
                   dashes=False, markers=True,
                   kind="line", data=s)

plt.show()
