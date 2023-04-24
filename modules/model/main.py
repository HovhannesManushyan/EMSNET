import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class ConvCatNet(nn.Module):
    def __init__(self, image_size=28):
        super(ConvCatNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=2)

        # First fully connected layer
        self.fc1 = nn.Linear(image_size*image_size+image_size*image_size, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = torch.flatten(self.conv1(x), 1)
        x2 = torch.flatten(self.conv2(x), 1)

        x = torch.cat([x1, x2], dim=1)
        # x = x1
        x = self.fc1(x)
        x = self.lrelu(x)

        x = self.fc2(x)

        output = torch.sigmoid(x)
        return output


def model_train_loop(net, train_dataloader, optimizer, criterion, bauc, epoch):
    running_loss = 0.0
    predictions = []
    ground_truth = []

    for idx, data in enumerate(train_dataloader, 0):
        # get the inputs
        # print(i,data)
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs.flatten(), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)

        optimizer.step()
        running_loss += loss.item()

        predictions.append(outputs.flatten())
        ground_truth.append(labels.flatten())

    # net.to('cpu')
    # plot_grad_flow(net.named_parameters())
    # plt.show()
    # net.to(device)

    predictions = torch.cat(predictions)
    ground_truth = torch.cat(ground_truth)

    healthy_f1 = f1_score((predictions > 0.5).float().cpu(), ground_truth.cpu(), pos_label=0)
    rocauc = bauc(predictions, ground_truth)
    print(
        '[%d, %5d] loss: %.3f, ROC: %.3f, F1_healthy: %.3f\n' % (epoch + 1, idx + 1, running_loss, rocauc, healthy_f1))


def model_test_loop(net, test_dataloader, criterion, bauc, epoch):
    with torch.no_grad():
        running_loss = 0.0
        test_predictions = []
        test_ground_truth = []

        for idx, data in enumerate(test_dataloader, 0):
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs.flatten(), labels)
            running_loss += loss.item()

            test_predictions.append(outputs.flatten())
            test_ground_truth.append(labels.flatten())

        test_predictions = torch.cat(test_predictions)
        test_ground_truth = torch.cat(test_ground_truth)

        healthy_f1 = f1_score((test_predictions > 0.5).float().cpu(), test_ground_truth.cpu(), pos_label=0)
        rocauc = bauc(test_predictions, test_ground_truth)
        print('[%d, %5d] Validation loss: %.3f, ROC: %.3f, F1_healthy: %.3f \n' % (
            epoch + 1, idx + 1, running_loss, rocauc, healthy_f1))
