import pandas as pd
import torch
import torch.nn as nn
import numpy as np

df = pd.read_csv("data/dow_jones_sentiment_top.csv")

data = df.iloc[:, 1:8].values
data = torch.tensor(data, dtype=torch.float32)

df['Close'] = pd.to_numeric(df['Close'], errors='coerce', downcast='float')
target = df['Close'].values
target = torch.tensor(target, dtype=torch.float32) 

data = torch.unsqueeze(data, dim=0)
target = torch.unsqueeze(target, dim=0)

# Define hyperparameters
input_size = 7
hidden_size = 7
sequence_length = data.shape[1]
learning_rate = 0.1
epochs = 40000

# Create the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # Defining layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out).squeeze(-1)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden

model = RNN(input_size, hidden_size, 1)
# model.load_state_dict(torch.load("rnn.pt"))

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
	outputs, _ = model(data)
	loss = criterion(outputs, target)

	optimizer.zero_grad()  # Zero the gradients
	loss.backward()  # Backpropagate
	optimizer.step()  # Update the weights

	if (epoch + 1) % 100 == 0:
		print(f'Epoch [{epoch + 1}/{epochs}], RMSE Loss: {np.sqrt(loss.item()):.4f}')

torch.save(model.state_dict(), "model/rnn.pt")

with torch.no_grad():
	_, final_embedding = model(data)

final_embedding = torch.squeeze(final_embedding, dim=0)
df_rnn = pd.DataFrame(data=final_embedding, columns=df.columns[1:8])
df_rnn.to_csv("data/dow_jones_sentiment_top_rnn.csv")