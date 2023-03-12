import torch
from tqdm import tqdm


def train_fnns(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterions,
    epochs,
    device="cpu",
):
    """
    Train a list feedforward neural networks, each is trained to predict one label in the output
    """
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader)
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            losses = [
                criterion(x_, torch.unsqueeze(y_, 1))
                for criterion, x_, y_ in zip(criterions, model(x), y.T)
            ]
            for loss in losses:
                loss.backward()

            optimizer.step()
            avg_loss = sum(losses) / len(losses)

            # validation
            model.eval()
            avg_val_loss = torch.tensor(0.0)
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_losses = [
                        criterion(x_, torch.unsqueeze(y_, 1))
                        for criterion, x_, y_ in zip(criterions, model(x), y.T)
                    ]
                    avg_val_loss += sum(val_losses) / (
                        len(val_losses) * len(val_loader)
                    )

            loop.set_description(
                f"Epoch: {epoch}, average_train_loss: {round(avg_loss.item(), 3)} average_val_loss: {round(avg_val_loss.item(), 3)}"
            )


def classifiers_fit(classifiers, x_train, y_train):
    """Fit a list of classifiers each to classify one label in the output"""
    for clf, y_label in zip(classifiers, y_train.T):
        clf.fit(x_train, y_label)
