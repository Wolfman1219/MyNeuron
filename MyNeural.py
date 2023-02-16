import numpy as np
import json
from activations import Activations


def MSE_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSE_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true.T) / y_true.size

class Neural(Activations):
    def __init__(self):
        self.weights = []
        self.biases = []
        self.loss = 'MSE'
        self.layers = []
    
    def add(self, neurons, activation):
        self.layers.append({'neurons': neurons, 'activation': activation})
    
    def random_params(self):
        for i in range(1, len(self.layers)):
            self.weights.append(2 * np.random.random((self.layers[i-1]['neurons'], self.layers[i]['neurons'])) - 1)
            self.biases.append(np.zeros((1, self.layers[i]['neurons'])))
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        self.random_params()
        
        for epoch in range(epochs):
            #forward
            activations = [X]
            
            for i in range(len(self.layers)-1):
                activations.append(self.activator(sample=(np.dot(activations[-1], self.weights[i]) + self.biases[i]), activator_name=self.layers[i]['activation']) )
            
            # birinchi error
            output = activations[-1]
            error = MSE_loss_derivative(y, output)
            deltas = error * self.activator(sample = output, activator_name=self.layers[-1]['activation'], deriative=True)
            
            # Backward 
            for i in range(len(activations)-2, -1, -1):
                grad_w = np.dot(activations[i].T, deltas)
                grad_b = np.sum(deltas, axis=0, keepdims=True)
                self.weights[i] -= learning_rate * grad_w
                self.biases[i] -= learning_rate * grad_b
                deltas = np.dot(deltas, self.weights[i].T) * self.activator(sample=(activations[i]), activator_name=self.layers[i]['activation'], deriative=True)
            
            
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