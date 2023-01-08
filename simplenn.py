import torch
import torch.nn as nn

neuronsIn, neuronsHidden, neuronsOut, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, neuronsIn)
y = torch.tensor([[1.0],[0.0],[0.0],[1.0],[1.0],[1.0],[0.0],[0.0],[1.0],[1.0]])

model = nn.Sequential(nn.Linear(neuronsIn, neuronsHidden),
                      nn.ReLU(),
                      nn.Linear(neuronsHidden, neuronsOut),
                      nn.Sigmoid())

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(50):
    y_pred = model(x)

    loss= criterion(y_pred, y)
    print("epoch: ", epoch, "loss: ", loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
