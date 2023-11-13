import json
from utils import tokenize, stem, bag_of_words
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import BotModel
from pathlib import Path

# Create dataset

class ChatbotDataset(Dataset):
    def __init__(self, path, transformer=None):
        self.transformer = transformer
        self.x, self.y, self.classes, self.all_tokens = self.load_data(path)
        self.n_samples = len(self.x)


    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def load_data(self, path):
        # Setup dataset and preprocessing
        with open(path, "r") as f:
            intents = json.load(f)

        all_tokens, tags, patterns_with_tag, ignore = [], [], [], ['?', '!', '.', ',']

        for intent in intents["intents"]:
            tag = intent["tag"]
            tags.append(tag)
            for pattern in intent["patterns"]:
                tokenized_pattern = tokenize(pattern)
                all_tokens.extend(tokenized_pattern)
                patterns_with_tag.append((tokenized_pattern, tag))

        tags = sorted(tags)
        all_tokens = [stem(token) for token in sorted(set(all_tokens)) if token not in ignore]

        X_train, y_train = [], []

        for tokenized_pattern, tag in patterns_with_tag:
            # Add new pattern data
            bag = bag_of_words(tokenized_pattern, all_tokens)
            X_train.append(bag)

            # Add new pattern label
            label = tags.index(tag)
            y_train.append(label)

        if self.transformer:
            return self.transformer(X_train), self.transformer(y_train), tags, all_tokens
        else:
            return X_train, y_train, tags, all_tokens


dataset = ChatbotDataset("intents.json")

# Define hyperparameters
MODEL_PATH = Path("./model.pt")
BATCH_SIZE = 8
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(dataset.classes)
INPUT_SIZE = len(dataset.x[0])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.001
EPOCHS = 1000

# Data loader
train_loader = DataLoader(dataset=dataset, batch_size = BATCH_SIZE, shuffle=True)

# Initialize Model
model = BotModel(INPUT_SIZE,HIDDEN_SIZE, OUTPUT_SIZE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)


if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH))
    EPOCHS = 10


# Training
for epoch in range(EPOCHS):
    # Training environment
    model.train()

    # Batch
    for (words, labels) in train_loader:
        words = words.to(DEVICE)
        labels = labels.to(dtype=torch.long).to(DEVICE)
        # Predict
        outputs = model(words)

        # Loss
        loss = criterion(outputs, labels)

        # Initialize grads
        optimizer.zero_grad()

        # Back propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch + 1} / {EPOCHS}, loss = {loss.item():.4f}")


# Save entire model

model_data = {
"model_state": model.state_dict(),
"input_size": INPUT_SIZE,
"hidden_size": HIDDEN_SIZE,
"output_size": OUTPUT_SIZE,
"all_tokens": dataset.all_tokens,
"tags": dataset.classes
}

torch.save(model_data, MODEL_PATH)