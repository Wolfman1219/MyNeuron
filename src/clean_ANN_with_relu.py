import numpy as np

# Define the neural network architecture
input_dim = 4
hidden_dim1 = 3
hidden_dim2 = 2
output_dim = 1

# Initialize the weights and biases randomly
np.random.seed(1)
w1 = np.ones((input_dim, hidden_dim1))
b1 = np.ones((1, hidden_dim1))
w2 = np.ones((hidden_dim1, hidden_dim2))
b2 = np.ones((1, hidden_dim2))
w3 = np.ones((hidden_dim2, output_dim))
b3 = np.ones((1, output_dim))


print("w1",w1.shape)
print(w1)
print("b1",b1.shape)
print(b1)

print("w2",w2.shape)
print(w2)
print("b2",b2.shape)
print(b2)

print("w3",w3.shape)
print(w3)
print("b3",b3.shape)
print(b3)


# Define the activation function (ReLU)
def ReLU(x):
    return np.maximum(0, x)

# Define the derivative of the activation function
def ReLU_derivative(x):
     return np.where(x > 0, 1, 0)

# Define the learning rate
learning_rate = 0.1

# Define the input data
X = np.array([[2, 3, 4, 5], [3, 4, 5, 6]])
y = np.array([[12, 14]])
print("shapelar:", X.shape, "  ", w1.shape)
# Train the neural network
for i in range(1):
    # Feedforward
    print("SHuni ko'r:", np.dot(X, w1))
    layer1 = ReLU(np.dot(X, w1) + b1)
    print(layer1)
    print("Birinchi layerning shape i",layer1.shape)
    layer2 = ReLU(np.dot(layer1, w2) + b2)
    print("2-layer:",layer2)
    layer3 = np.dot(layer2, w3) + b3
    print(layer3)

    # Backpropagation
    layer3_error = y - layer3
    print("3-layerni errori:",layer3_error)
    layer3_delta = layer3_error
    layer2_error = layer3_delta.dot(w3.T)
    print("layer2_error:",layer2_error)
    layer2_delta = layer2_error * ReLU_derivative(layer2)
    print("ReLU hosila:", ReLU_derivative(layer2))
    print("dl2:",layer2_delta)
    layer1_error = layer2_delta.dot(w2.T)
    print('l1 error:', layer1_error)
    layer1_delta = layer1_error * ReLU_derivative(layer1)
    print("dl1",layer1_delta)


    # Update the weights and biases
    print("bir nimaal:", layer2.T.dot(layer3_delta))
    w3 -= learning_rate * layer2.T.dot(layer3_delta)
    print("w3:", w3)

    b3 -= learning_rate * np.sum(layer3_delta, axis=0, keepdims=True)
    print("b3:", b3)

    w2 -= learning_rate * layer1.T.dot(layer2_delta)
    print("w2:", w2)

    b2 -= learning_rate * np.sum(layer2_delta, axis=0, keepdims=True)
    print("b2:", b2)

    w1 -= learning_rate * X.T.dot(layer1_delta)
    print("w1:", w1)

    b1 -= learning_rate * np.sum(layer1_delta, axis=0, keepdims=True)
    print("b1:", b1)



# Test the neural network
# test_data = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]])
# test_output = np.dot(ReLU(np.dot(ReLU(np.dot(ReLU(np.dot(test_data, w1) + b1), w2) + b2), w3) + b3), np.ones((output_dim, 1)))
# print(test_output)
