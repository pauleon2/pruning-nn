import torch


def test(test_loader, model):
    """
    Test the model on the test dataset provided by the test loader.
    :param test_loader: The provided test dataset
    :param model: The model that should be tested
    :return: The percentage of correctly classified samples from the dataset.
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for test_images, test_labels in test_loader:
            test_images = test_images.reshape(-1, 28 * 28)
            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        per = 100 * correct / total
        print('Accuracy of the network on the {} test images: {:.4f} %'
              .format(total, per))
        return per


def train(train_loader, model, optimizer, criterion, epoch, total_epochs):
    """
    Train the model on the train dataset with a loss function and and optimization algorithm.
    :param train_loader: The training dataset.
    :param model: The to be trained model
    :param optimizer: The used optimizer.
    :param criterion: The loss function.
    :param epoch: The current epoch.
    :param total_epochs: The total number of epochs.
    :return: The average loss of the network in this epoch.
    """
    total_step = len(train_loader)
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, total_epochs, i + 1, total_step, loss.item()))
    return total_loss / len(train_loader)
