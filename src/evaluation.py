import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def evaluate(y_true, y_pred):
    """Evaluate the model on the test set.
    y_true, y_pred: are one-hot-encoded labels
    """
    assert len(y_true) == len(y_pred)
    print(
        "Kind accuracy: ", (np.array(y_true) == np.array(y_pred)).mean()
    )  # consider each label individually, y_pred = [0, 0, 1], y_true = [0, 0, 0] gets +2 and -1
    print(
        "Harsh accuracy: ", accuracy_score(y_true, y_pred)
    )  # consider all labels together, if one label is wrong, the whole prediction is wrong
    print("Classification Report: ")
    print(classification_report(y_true, y_pred))
