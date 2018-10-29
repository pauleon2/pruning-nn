import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruninig_nn.network import NeuralNetwork

hyper_params = {
    'pruning_percentage': 90.,  # percentage of weights pruned
    'batch_size': 64,
    'test_batch_size': 100,
    'num_epochs': 10,
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
criterion = nn.NLLLoss()  # TODO: actually use MSE Error
optimizer = torch.optim.SGD(model.parameters(),
                            lr=hyper_params['learning_rate'],
                            momentum=hyper_params['momentum'])

total_step = len(train_loader)
for epoch in range(hyper_params['num_epochs']):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, hyper_params['num_epochs'], i + 1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for test_images, test_labels in test_loader:
            test_images = test_images.reshape(-1, 28 * 28)
            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

        print('Accuracy of the network on the {} test images: {:.4f} %'
              .format(total, 100 * correct / total))

# TODO: do pruning process here

# TODO: reevaluate test accuracy

# TODO: do retraining

# TODO: reevaluate test accuracy
