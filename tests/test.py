import torch

from src.model import FNN
from src.train import train_fnn


def test_fnn():
    """
    Test FNN
    """
    fnn = FNN(input_dim=10, hidden_dim=10, number_of_labels=5)
    assert fnn is not None
    assert len(fnn(torch.rand(10))) == 5
