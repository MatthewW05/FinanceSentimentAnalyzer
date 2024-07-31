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
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.optim as optim
import pickle

# Constants
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 16

# Utility functions
def detect_encoding(file_path):
    import chardet
    with open(file_path, 'rb') as file:
        sample = file.read(10000)  # Read only a sample for efficiency
        result = chardet.detect(sample)
        return result['encoding']

def clean_data(file_info, output_file):
    """Clean data from specified files and save to a new CSV file, handling different encodings."""
    # Initialize the output file as blank
    open(output_file, 'w').close()

    fields = ['headline', 'sentiment']
    all_data = []

    for info in file_info:
        file_path = info[0]
        title_index = info[1]
        sentiment_index = info[2]
        encoding = detect_encoding(file_path)
        global test
        global t
        
        with open(file_path, mode='r', encoding=encoding)as file:
            if file_path.endswith('.csv'):
                csvFile = csv.reader(file)

                for line in csvFile:
                    s = convert_sentiment(line[sentiment_index])
                    h = re.sub(r'[^A-Za-z0-9%., ]+', '', line[title_index])
                    h = h.replace('%', 'percent')
                    if type(s) == int:
                        all_data.append([h, s])
            
            elif file_path.endswith('.txt'):
                txtFile = file.readlines()

                for line in txtFile:
                    l = line.split('@')
                    
                    s = convert_sentiment(l[sentiment_index])
                    h = re.sub(r'[^A-Za-z0-9%., ]+', '', l[title_index])
                    h = h.replace('%', 'percent')
                    if type(s) == int:
                        all_data.append([h, s])
    
    df = pd.DataFrame(all_data, columns=fields)
    df.to_csv(output_file, index=False)
    
    return output_file

def convert_sentiment(value):
    """Convert sentiment values to numeric format."""
    if value.strip().lower() == 'positive' or value.strip() == '1':
        return 1
    elif value.strip().lower() == 'negative' or value.strip() == '0':
        return 0
    else:
        try:
            temp = json.loads(value)
            pos = 0
            neg = 0

            for key in temp:
                if temp[key].strip().lower() == 'positive':
                    pos += 1
                elif temp[key].strip().lower() == 'negative':
                    neg += 1
                
            if pos - neg > 0:
                return 1
            elif pos-neg < 0:
                return 0
        except:
            pass
    
    return None

# PyTorch Dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        # Tokenize the text
        tokenized_text = [self.vocab[token] for token in self.tokenizer(text)]
        # Convert to tensor
        text_tensor = torch.tensor(tokenized_text, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return text_tensor, label_tensor

# Model Definition using PyTorch
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

# Function to create a vocab and tokenizer
def build_vocab(headlines, tokenizer):
    vocab = build_vocab_from_iterator(map(tokenizer, headlines), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# Function to preprocess and tokenize text
def preprocess_and_tokenize(data, tokenizer, vocab):
    tokenized = [torch.tensor(vocab(tokenizer(item[0])), dtype=torch.int64) for item in data]
    labels = torch.tensor([item[1] for item in data], dtype=torch.float32)
    return tokenized, labels

def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(texts)  # Forward pass
            
            # Compute loss
            outputs = outputs.squeeze()  # Adjust the output dimensions if necessary
            loss = criterion(outputs, labels.float())
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()  # Assuming the output is in the range [0,1] and using 0.5 as threshold
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Compute average loss and accuracy
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples * 100

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def save_vocab(vocab, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

def save_model(model, model_path, vocab, vocab_path):
    torch.save(model.state_dict(), model_path)
    save_vocab(vocab, vocab_path)

def load_vocab(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_model_and_vocab(model_path, vocab_path, embedding_dim):
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    model = SentimentModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model, vocab

def evaluate_model(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            predicted = outputs.round()  # Assuming sigmoid output
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Loss: {total_loss/len(data_loader)}, Accuracy: {accuracy}%')
    return accuracy

def predict_headline_sentiment(headline, model, vocab, tokenizer):
    model.eval()  # Evaluation mode
    tokens = torch.tensor([vocab[token] for token in tokenizer(headline)], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tokens)
        prediction = output.item()
    return prediction

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.float32)
    return text_list.to(device), labels.to(device)

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    file_info = [
        [f'{dir_path}\\raw_data\\all-data.csv', 1, 0], 
        [f'{dir_path}\\raw_data\\data.csv', 0, 1], 
        [f'{dir_path}\\raw_data\\Fin_Cleaned.csv', 1, 4],
        [f'{dir_path}\\raw_data\\Sentences_75Agree.txt', 0, 1],
        [f'{dir_path}\\raw_data\\SEntFiN-v1.1.csv', 1, 2],
    ]

    output_file = f'{dir_path}\\cleaned_data.csv'
    clean_data(file_info, output_file)

    df = pd.read_csv(output_file)
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    # Building vocab and tokenizer
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab(df_train['headline'].tolist(), tokenizer)
    vocab_size = len(vocab)

    train_dataset = SentimentDataset(df_train, tokenizer, vocab)
    test_dataset = SentimentDataset(df_val, tokenizer, vocab)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = SentimentModel(vocab_size, EMBEDDING_DIM)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    print("Starting training...")
    num_epochs = 30
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluating the model
    print("Starting evaluation...")
    accuracy = evaluate_model(model, test_loader, criterion)
    accuracy = str(round(accuracy, 1)).split('.')
    name = f'{datetime.now()}--'.strip().replace(' ', '-').replace(':', '-').split('.')
    name = name[0] + '--' + accuracy[0] + '_' + accuracy[1]
    print(f"Test accuracy: {accuracy[0]}.{accuracy[1]}%")

    # Saving the model
    model_path = f'{dir_path}\\FinanceSentimentAnalyzer\\trained_models\\{name}.pth'
    vocab_path = f'{dir_path}\\FinanceSentimentAnalyzer\\trained_models\\{name}.pkl'
    save_model(model, model_path, vocab, vocab_path)
    print(f"Model saved to {model_path}")

    # Loading the model
    loaded_model, vocab = load_model_and_vocab(model_path, vocab_path, EMBEDDING_DIM)
    loaded_model.to(device)

    # Example prediction
    sample_headline = "Nvidia Stock Rises. How Earnings From Microsoft and Apple Could Drive It Higher."
    prediction = predict_headline_sentiment(sample_headline, loaded_model, vocab, tokenizer)
    prediction = "Positive" if round(prediction) == 1 else "Negative"

    print(f"Prediction for \'{sample_headline}\': {prediction}")