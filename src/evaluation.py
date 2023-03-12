from sklearn.metrics import (accuracy_score, classification_report)
import numpy as np

def evaluate(y_true, y_pred):
    """Evaluate the model on the test set.
    y_true, y_pred: are one-hot-encoded labels
    """
    
    print("Kind accuracy: ", (np.array(y_true) == np.array(y_pred)).mean()) # consider each label individually, y_pred = [0, 0, 1], y_true = [0, 0, 0] gets +2 and -1  
    print("Harsh accuracy: ", accuracy_score(y_true, y_pred)) # consider all labels together, if one label is wrong, the whole prediction is wrong
    print("Classification Report: ")
    print(classification_report(y_true, y_pred))

def evaluate_gpt(response_list:list[str], y_true, label_encoder):
    """
    Evaluate the model on the test set.
    responses: [[predicted topic1, predicted topic2, ...], [predicted topic1, predicted topic2, ...], ...]
    y_true: are the true one-hot-encoded  labels
    one_hot_encoder: is the one-hot encoder used to encode the labels"""
    
    y_pred = [label_encoder.transform([response]) for response in response_list]
    evaluate(y_true, y_pred)

