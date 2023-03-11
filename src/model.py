import os

import numpy as np
import openai
import torch.nn as nn
from dotenv import load_dotenv

# env variables
load_dotenv(dotenv_path="../.env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text):
    """
    Get the embedding of a text using ada-002
    """
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


def GPT_3(prompt):
    """
    Get the response of GPT-3
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["]"],
    )
    return "[" + response["choices"][0]["text"] + "]"  # add brackets to make it a list


class DivideAndConquerFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_labels):
        """
        A list of feedforward neural network, each is trained to predict one label in the output
        """
        super(DivideAndConquerFNN, self).__init__()
        self.seq = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(number_of_labels)
            ]
        )

    def forward(self, x):
        return [classifier(x) for classifier in self.seq]


def classifiers_fit(classifiers, x_train, y_train):
    """Fit a list of classifiers each to classify one label in the output"""
    assert len(classifiers) == y_train.shape[1]
    for clf, y_label in zip(classifiers, y_train.T):
        clf.fit(x_train, y_label)


def classifiers_predict(classifiers, x_test):
    """Predict the output of a list of classifiers"""
    predictions = []
    for clf in classifiers:
        predictions.append(clf.predict(x_test))
    return np.array(predictions).T
