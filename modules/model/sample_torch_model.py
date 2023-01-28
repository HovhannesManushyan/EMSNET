import torch
import torch.nn as nn


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

    def train(self, data, y, data_test, y_test, loss_fn=nn.BCELoss(), batch_size=10, epochs=4, optimizer=torch.optim.Adam):
        optimizer = optimizer(self.parameters())
        data = torch.split(data, batch_size)
        y = torch.split(y, batch_size)

        for epoch in range(epochs):
            for i in range(len(data)):
                optimizer.zero_grad()
                output = self(data[i])
                loss = loss_fn(output, y[i])
                loss.backward()
                optimizer.step()
                accuracy = (output.round() == y[i]).float().mean()
                print('Epoch {} Loss {} Accuracy {}'.format(epoch, loss.item(), accuracy))

        with torch.no_grad():
            output_test = self(data_test)
            accuracy_test = (output_test.round() == y_test).float().mean()
            print('Test Accuracy:', accuracy_test.item())
