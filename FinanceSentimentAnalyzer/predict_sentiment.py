import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import csv
import json
import re
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.optim as optim
import pickle
import logging
import warnings

MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 16

__dir_path = os.path.dirname(os.path.realpath(__file__))
__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 16, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last time step's outputs
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def load_vocab(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_model_and_vocab(model_path, vocab_path, embedding_dim=EMBEDDING_DIM):
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    model = SentimentModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(__device)))
    model.eval()  # Set the model to evaluation mode
    return model, vocab

__model_path = f'{__dir_path}\\trained_models\\sentiment_analysis.pth'
__vocab_path = f'{__dir_path}\\trained_models\\sentiment_analysis.pkl'
__default_model, __default_model_vocab = load_model_and_vocab(__model_path, __vocab_path, EMBEDDING_DIM)
__default_model.to(__device)

__default_tokenizer = get_tokenizer('basic_english')

def predict_headline_sentiment(headline, model=__default_model, vocab=__default_model_vocab, tokenizer=__default_tokenizer):
    model.eval()  # Evaluation mode
    tokens = torch.tensor([vocab[token] for token in tokenizer(headline)], dtype=torch.long).unsqueeze(0).to(__device)
    with torch.no_grad():
        output = model(tokens)
        prediction = output.item()
    return prediction


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading the model
    '''
    model_path = f'{dir_path}\\trained_models\\sentiment_analysis.pth'
    vocab_path = f'{dir_path}\\trained_models\\sentiment_analysis.pkl'
    loaded_model, vocab = load_model_and_vocab(model_path, vocab_path, EMBEDDING_DIM)
    loaded_model.to(device)

    tokenizer = get_tokenizer('basic_english')
    '''

    # Example prediction
    sample_headline = "Nvidia Stock Rises. How Earnings From Microsoft and Apple Could Drive It Higher."
    prediction = predict_headline_sentiment(sample_headline)
    prediction = "Positive" if round(prediction) == 1 else "Negative"

    print(f"Prediction for \'{sample_headline}\': {prediction}")
