#from google.colab import drive
#drive.mount('/content/drive')

#%cd /content/drive/MyDrive/class/sp_2025/prog_sp_10
#!pwd
#!ls

# prog_sp_10_pytorch.py
# layer 3 perceptron, hidden size 5, Relu, cross-entropy error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#pytorch check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Fix the seed of random number generator for reproducibility
#torch.manual_seed(113)
#np.random.seed(113)
torch.manual_seed(100)
np.random.seed(100)

# Data generation
N = 100000
x0 = np.random.uniform(-2, 2, N)
x1 = np.random.uniform(-2, 2, N)
X = np.column_stack((x0, x1))

# Label(correct answer) generation
y = np.zeros(N)
y[(x0 >= 0) & (x1 >= 0)] = 0
y[(x0 < 0) & (x1 >= 0)] = 1
y[(x0 < 0) & (x1 < 0)] = 0
y[(x0 >= 0) & (x1 < 0)] = 1

# Separate data into traing data and evaluation data
X_train, X_eval = X[:80000], X[80000:]
y_train, y_eval = y[:80000], y[80000:]

# Data transform to Tensor format 
X_train = torch.tensor(X_train, dtype=torch.float32)
X_eval = torch.tensor(X_eval, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_eval = torch.tensor(y_eval, dtype=torch.long)

# Model definition
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

model = Perceptron()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training (learning)
epochs = 100
batch_size = 64
train_losses = []
eval_accuracies = []

#measure end time (optional)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#--- main loop
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        X_batch, y_batch = X_train[indices], y_train[indices]

        # forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # back-propagation and parameter updating
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        # save training loss 
        train_losses.append(loss.item())

        # evaluation
        with torch.no_grad():
            outputs_eval = model(X_eval)
            _, predicted = torch.max(outputs_eval, 1)
            accuracy = (predicted == y_eval).float().mean().item()
            eval_accuracies.append(accuracy)

        print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}')

#measure end time (optional)
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
print(elapsed_time / 1000, 'sec.')


#--- Display training loss and evaluation accuracy
print("Training Losses:", train_losses)
print("Evaluation Accuracies:", eval_accuracies)

# plot training loss and evaluation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(0, epochs, 10), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(0, epochs, 10), eval_accuracies, label='Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('pytorch_loss_accuracy_plot.png')
plt.show()

# plot circles in (x0, x1) plane. if y >= 0.5 circle, color is red, others blue.
with torch.no_grad():
    outputs_eval = model(X_eval)
    y_pred_eval = torch.softmax(outputs_eval, dim=1).numpy()

plt.figure(figsize=(6, 6))
plt.scatter(X_eval[:, 0], X_eval[:, 1], c=(y_pred_eval[:, 1] >= 0.5), cmap='coolwarm', marker='o', alpha=0.5)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Points with y >= 0.5')
plt.savefig('pytorch_predicted_plot.png')
plt.show()
