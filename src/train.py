import torch
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    weights,
    epochs,
    device="cpu",
):
    """
    Train a list feedforward neural networks, each is trained to predict one label in the output
    the training criterion is the sum of the weighted losses of each classifier
    """
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = sum(
                [
                    w * criterion(x_, torch.unsqueeze(y_, 1))
                    for w, x_, y_ in zip(weights, model(x), y.T)
                ]
            )

            loss.backward()
            optimizer.step()

            # validation
            val_loss = 0
            model.eval()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += sum(
                    [
                        w * criterion(x_, torch.unsqueeze(y_, 1))
                        for w, x_, y_ in zip(weights, model(x), y.T)
                    ]
                )
            val_loss /= len(val_loader)

            loop.set_description(
                f"Epoch: {epoch}, Train_loss: {round(loss.item(), 3)} Val_loss: {round(val_loss.item(), 3)}"
            )
