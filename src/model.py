import os

import openai
import torch
import torch.nn as nn
from dotenv import load_dotenv

# env
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
    return response["choices"][0]["text"]


class DivideAndConquerFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_labels):
        """
        Feedforward neural network
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
