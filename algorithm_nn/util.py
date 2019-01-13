import torch


def test(test_loader, model):
    """
    Test the model on the test data set provided by the test loader.

    :param test_loader:     The provided test data set
    :param model:           The model that should be tested
    :return: The percentage of correctly classified samples from the data set.
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
        return 100 * correct / total


def train(train_loader, model, optimizer, criterion, epoch, total_epochs, percentage=False):
    """
    Train the model on the train data set with a loss function and and optimization algorithm.

    :param train_loader:    The training data set.
    :param model:           The to be trained model.
    :param optimizer:       The used optimizer.
    :param criterion:       The loss function.
    :param epoch:           The current epoch.
    :param total_epochs:    The total number of epochs.
    :param percentage:      If the function should also calculate the percentage of right made decisions
    :return: The average loss of the network in this epoch.
    """
    total_step = len(train_loader)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        if percentage:
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

    if percentage:
        per = 100 * correct / total
    else:
        per = 0

    return total_loss / len(train_loader), per
