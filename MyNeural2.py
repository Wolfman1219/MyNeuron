import numpy as np
import json
from activations import Activations
import time

def MSE_loss(y_true, y_pred):
    return np.mean((y_true.T - y_pred) ** 2)

def MSE_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true.T) / y_true.size

class Neural(Activations):
    def __init__(self):
        self.weights = []
        self.biases = []
        self.loss = 'MSE'
        self.layers = []
        self.h = []
    
    def add(self, neurons, activation):
        self.layers.append({'neurons': neurons, 'activation': activation})
    
    def random_params(self):
        for i in range(1, len(self.layers)):
            self.weights.append(2 * np.random.random((self.layers[i-1]['neurons'], self.layers[i]['neurons'])) - 1)
            self.biases.append(np.zeros((1, self.layers[i]['neurons'])))
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        self.random_params()
        
        for epoch in range(epochs):
            batch_indices = np.random.permutation(len(X))
            # print(batch_indices)
            batches_X = np.array_split(X, len(X) // batch_size)
            batches_y = np.array_split(y[0], len(X) // batch_size)
            
            for batch_X, batch_y in zip(batches_X, batches_y):
                activations = [batch_X]
                
                for i in range(1, len(self.layers)):
                    activations.append(self.activator(sample = np.dot(activations[-1], self.weights[i-1]) + self.biases[i-1], activator_name=self.layers[i]['activation']))
                
                output = activations[-1]
                # print(output.shape)
                # print(batch_y.shape)
                # print(batch_X.shape)

                deltas = MSE_loss_derivative(batch_y, output)
                
                for i in range(len(activations)-1, 0, -1):
                    self.weights[i-1] = self.weights[i-1] - learning_rate * activations[i-1].T.dot(deltas)
                    self.biases[i-1] = self.biases[i-1] - learning_rate * np.sum(deltas, axis=0, keepdims=True)
                    deltas = deltas.dot(self.weights[i-1].T) * self.activator(activator_name = self.layers[i]["activation"],derivative = True, sample = activations[i-1])
                    
            if epoch % 100 == 0:
                time.sleep(0.5)
                loss = np.mean([MSE_loss(self.predict(batch_X), batch_y) for batch_X, batch_y in zip(batches_X, batches_y)])
                text = f"Epoch: {epoch}  loss: {loss}"
                print("\r" + " " * len(text) + "\r", end="", flush=True)
                print(text, end="",flush=True)
                
        print("\n")

               
    def predict(self,X):
        self.h = [X]
        for i in range(1, len(self.layers)):
                self.h.append(self.activator(sample = np.dot(self.h[-1], self.weights[i-1]) + self.biases[i-1], activator_name=self.layers[i]['activation']))
        return self.h[-1]

    def save_model(self, model_path):
        model_data = {}
        model_data['layers'] = self.layers
        model_data['weights'] = [w.tolist() for w in self.weights]
        model_data['bias'] = [b.tolist() for b in self.biases]
        model_data['loss'] = self.loss
        with open(model_path, 'w') as outfile:
            json.dump(model_data, outfile)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path) as json_file:
            model_data = json.load(json_file)
        model = cls()
        model.layers = model_data['layers']
        model.weights = [np.array(w) for w in model_data['weights']]
        model.bias = [np.array(b) for b in model_data['bias']]
        model.loss = model_data['loss']
        return model

def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true