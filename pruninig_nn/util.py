import torch


def test(test_loader, model):
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


def train(train_loader, model, optimizer, criterion, epoch, total_epochs):
    total_step = len(train_loader)
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
                  .format(epoch + 1, total_epochs, i + 1, total_step, loss.item()))
