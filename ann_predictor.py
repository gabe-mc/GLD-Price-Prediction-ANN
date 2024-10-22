"""Predict the price of GLD using an artifical neural network"""

import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Splitting into testing and training data

data = pd.read_csv("data/training_data.csv")
data = data.drop("Date", axis=1)

data = data.sample(frac=1)

num_rows = data.shape[0]
train_split_percentage = 0.75
split = math.floor(num_rows*train_split_percentage)

testing_data = data[0:split]
training_data = data[split:]


# Preprocessing data

def min_max_scaler(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the array data into values between 0 and 1. Takes a panada DataFrame and creates a new panda DataFrame.
    """
    result = pd.DataFrame(columns=data.columns)
    for column in data:
        loop_index = 0
        for value in data[column].values:
            loop_index += 1
            column_max = data[column].max()
            column_min = data[column].min()
            new_value = (float(value) - float(column_min)) / \
                (float(column_max) - float(column_min))
            result.at[loop_index, column] = new_value

    return result


target_scaled = np.array(min_max_scaler(testing_data))
features_scaled = np.array(min_max_scaler(training_data))

testing_target = testing_data["GLD Price"]
testing_features = testing_data.drop("GLD Price", axis=1)

training_target = training_data["GLD Price"]
training_features = training_data.drop("GLD Price", axis=1)


# Creating ANN

learning_rate = 0.6


class GLDPredictor(nn.Module):
    """
    Artificial Neural Network predicting the price of $GLD.
    - Three layers of neruons
    - 
    """

    def __init__(self):
        super(GLDPredictor, self).__init__()
        self.layer1 = nn.Linear(9, 12)
        self.layer2 = nn.Linear(12, 12)
        self.output = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.output(x))
        return x


model = GLDPredictor()
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    # Initalize tensors
    inputs = torch.tensor(training_features.to_numpy(), dtype=torch.float32)
    labels = torch.tensor(training_target.to_numpy(), dtype=torch.float32)

    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, labels.unsqueeze(1))

    # Backwards pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}. Remaining: {epochs - epoch+1}. Loss: {loss.item():.4f}')
