"""Class creating and training the ANN model from time series data"""

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

from data_transformation import normalize, split
from src.tensor import Tensor

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
test_dataloader = D.DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False)

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
    
# Training GLDPredictor on our training data

model = GLDPredictor()

# # Instantiate our loss function

# learning_rate = 0.001
# loss_function = nn.L1Loss() # Mean absoloute error
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# # Our training loop

# epochs = 1000000
# loss_values = []

# for epoch in range(epochs):
#     for features, target in train_dataloader:
#         optimizer.zero_grad()

#         # Forward pass
#         prediction = model(features)
#         loss = loss_function(prediction, target.unsqueeze(-1))
#         loss_values.append(loss.item())

#         # Backward pass + optimization
#         loss.backward()
#         optimizer.step()

#         # Print out loss
#     if epoch % 10 == 0:
#         print(loss.item())

# print(f"Training Complete. Final loss: {loss_values[-1]}")

# Save the trained model

# torch.save(model.state_dict(), "models/GLDtrained2.plt")

# Validate our model

# def validate_model(model, val_dataloader):
    
#     losses = []
#     # Define the loss function
#     loss_function = nn.L1Loss()

#     # Track total loss
#     total_loss = 0.0
#     num_batches = 0

#     with torch.no_grad():
#         for features, target in val_dataloader:

#             # Forward pass
#             prediction = model(features)
#             loss = loss_function(prediction, target.unsqueeze(-1))
#             total_loss += 1
#             losses.append(loss.item())
#             num_batches += 1

#     # Average loss
#     avg_loss = total_loss / num_batches
#     print(f"Validation Complete. Average Loss: {avg_loss}")

#     return losses


# validation_loss = validate_model(model, test_dataloader)

# Plotting loss function

# x0 = list(range(1, len(validation_loss)+1))
# plt.figure(figsize=(5, 2))
# plt.plot(x0, validation_loss, label='Validation loss')
# plt.title('Validation loss')
# plt.legend()
# plt.show()