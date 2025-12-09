#from google.colab import drive
#drive.mount('/content/drive')

#%cd /content/drive/MyDrive/class/sp_2025/prog_sp_10
#!pwd
#!ls

#prog_sp_10_sigmoid.py 
#sigmoid, layer 3, hidden_size 5 ->20(better, but not necessary)
#
import numpy as np
import matplotlib.pyplot as plt

# Fix the seed of random number generator for reproducibility
#np.random.seed(42)
np.random.seed(118)

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
y = np.eye(2)[y.astype(int)]  # one-hotエンコーディング

# Separate data into traing data and evaluation data
X_train, X_eval = X[:80000], X[80000:]
y_train, y_eval = y[:80000], y[80000:]

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid function derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# cross entropy erro
def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

# parameters initialization
input_size = 2
hidden_size1 = 5
hidden_size2 = 5
output_size = 2

W1 = np.random.randn(input_size, hidden_size1)
b1 = np.random.randn(hidden_size1)
W2 = np.random.randn(hidden_size1, hidden_size2)
b2 = np.random.randn(hidden_size2)
W3 = np.random.randn(hidden_size2, output_size)
b3 = np.random.randn(output_size)

# learning rate
learning_rate = 0.01

# traing epochs
epochs = 100
batch_size = 64
train_losses = []
eval_accuracies = []

for epoch in range(epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # forward pass
        z1 = np.dot(X_batch, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, W3) + b3
        y_pred = softmax(z3)

        # loss calculation
        loss = cross_entropy_error(y_pred, y_batch)

        # back propagation
        delta3 = y_pred - y_batch
        dW3 = np.dot(a2.T, delta3) / X_batch.shape[0]
        db3 = np.sum(delta3, axis=0) / X_batch.shape[0]

        delta2 = np.dot(delta3, W3.T) * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, delta2) / X_batch.shape[0]
        db2 = np.sum(delta2, axis=0) / X_batch.shape[0]

        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(X_batch.T, delta1) / X_batch.shape[0]
        db1 = np.sum(delta1, axis=0) / X_batch.shape[0]

        # parameters updating
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    if epoch % 10 == 0:
        # save training loss
        train_losses.append(loss)

        # evaluation
        z1_eval = np.dot(X_eval, W1) + b1
        a1_eval = sigmoid(z1_eval)
        z2_eval = np.dot(a1_eval, W2) + b2
        a2_eval = sigmoid(z2_eval)
        z3_eval = np.dot(a2_eval, W3) + b3
        y_pred_eval = softmax(z3_eval)

        # calculate evaluation accuracy
        accuracy = np.mean(np.argmax(y_pred_eval, axis=1) == np.argmax(y_eval, axis=1))
        eval_accuracies.append(accuracy)

        print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

# Display training losses and evaluation accuracies
print("Training Losses:", train_losses)
print("Evaluation Accuracies:", eval_accuracies)

# plot training losses and evaluation accuracies
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
plt.show()

# plot circles in (x0, x1) plane. if y >= 0.5 circle, color is red, others blue.
plt.figure(figsize=(6, 6))
plt.scatter(X_eval[:, 0], X_eval[:, 1], c=(y_pred_eval[:, 1] >= 0.5), cmap='coolwarm', marker='o', alpha=0.5)
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Points with y >= 0.5')
plt.show()
