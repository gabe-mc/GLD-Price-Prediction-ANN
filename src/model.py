"""Class creating and training the ANN model from time series data"""

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

from data_transformation import normalize, split
from tensor import Tensor

import matplotlib.pyplot as plt
import numpy as np

# Import data

data = pd.read_csv("data/training_data.csv")
data = data.drop("Date", axis=1)

# Split and normalize data

training_data, testing_data = split(data, 0.75)
training_data = normalize(training_data)
testing_data = normalize(testing_data)

testing_target = testing_data["GLD Price"]
testing_features = testing_data.drop("GLD Price", axis=1)

training_target = training_data["GLD Price"]
training_features = training_data.drop("GLD Price", axis=1) 
    
# Convert DataFrames into Pytorch tensors, along with their respective Pytorch DataLoaders

batch_size = 64

train_tensor = Tensor(features=training_features.infer_objects(), target= training_target.infer_objects())
train_dataloader = D.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=False)
test_tensor = Tensor(features=testing_features.infer_objects(), target=testing_target.infer_objects())
test_dataloader = D.DataLoader(dataset=test_tensor, shuffle=False)

# Creating our prediction ANN

class GLDPredictor(nn.Module):
    """
    Artificial Neural Network predicting the price of $GLD. 
    Three layers of neruons, using affine linear transformations and ReLU activation functions.
    """

    def __init__(self):
        super(GLDPredictor, self).__init__()
        self.layer1 = nn.Linear(in_features=9, out_features=12, dtype=torch.float64)
        self.layer2 = nn.Linear(in_features=12, out_features=12, dtype=torch.float64)
        self.output = nn.Linear(in_features=12, out_features=1, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.output(x))

        return x
