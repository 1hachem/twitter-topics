import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.model import DivideAndConquerFNN
from src.train import classifiers_fit, train_fnns


def test_DivideAndConquerFNN():
    batch_size = 10
    input_dim = 20
    hidden_dim = 5
    number_of_labels = 3
    model = DivideAndConquerFNN(input_dim, hidden_dim, number_of_labels)
    x = torch.randn(batch_size, input_dim)
    y_pred = model(x)

    assert len(y_pred) == number_of_labels
    for y in y_pred:
        assert y.shape == (batch_size, 1)


def test_train_fnns():
    # Define the model architecture
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Define the loss criterion
    criterion = nn.MSELoss()

    # Define the training and validation data
    x_train = torch.randn(20, 10, 10)
    y_train = torch.randn(20, 1)

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=10)
    val_data = TensorDataset(x_train, y_train)
    val_loader = DataLoader(val_data, batch_size=10)

    # Train the model
    train_fnns(model, train_loader, val_loader, optimizer, [criterion], epochs=2)

    # Test the model
    assert isinstance(model, nn.Sequential)


def test_classifiers_fit():
    # Define the training data
    x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])

    # Define the classifiers
    classifiers = [LogisticRegression(random_state=0) for i in range(2)]

    # Fit the classifiers
    classifiers_fit(classifiers, x_train, y_train)

    assert isinstance(classifiers[0], LogisticRegression)


from src.utils import split_data


def test_split_data():
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100, 5))

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    assert x_train.shape == (72, 10)
    assert x_val.shape == (8, 10)
    assert x_test.shape == (20, 10)
    assert y_train.shape == (72, 5)
    assert y_val.shape == (8, 5)
    assert y_test.shape == (20, 5)
