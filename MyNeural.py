import numpy as np
import json
from activations import Activations
import time
import MyLoss


class Neural(Activations):
    def __init__(self, loss = MyLoss.MSE_loss):
        self.weights = []
        self.biases = []
        self.loss = loss
        self.layers = []
        self.h = []
    
    def add(self, neurons, activation):
        self.layers.append({'neurons': neurons, 'activation': activation})
    
    def random_params(self):
        for i in range(1, len(self.layers)):
            self.weights.append(2 * np.random.random((self.layers[i-1]['neurons'], self.layers[i]['neurons'])) - 1)
            self.biases.append(np.zeros((1, self.layers[i]['neurons'])))
    
    
    def fit(self, X, y, epochs=1000, batch_size = 32, learning_rate=0.01):
        self.random_params()
        batch_x = np.array_split(X, len(X)//batch_size)
        batch_y = np.array_split(y, len(X)//batch_size, axis=1)

        interval = epochs // 10
        interval2 = interval
        for epoch in range(epochs):
       
            for mini_x, mini_y in zip(batch_x, batch_y):

                activations = [mini_x]
                
                for i in range(1, len(self.layers)):
                    activations.append(self.activator(sample = np.dot(activations[-1], self.weights[i-1]) + self.biases[i-1], activator_name=self.layers[i]['activation']))
                
                output = activations[-1]
                deltas = self.loss(y_true=mini_y, y_pred=output, derivative = True)
                
                # Backward
                for i in range(len(activations)-1, 0, -1):
                    self.weights[i-1] = self.weights[i-1] - learning_rate * activations[i-1].T.dot(deltas)
                    self.biases[i-1] = self.biases[i-1] - learning_rate * np.sum(deltas, axis=0, keepdims=True)
                    deltas = deltas.dot(self.weights[i-1].T) * self.activator(activator_name = self.layers[i]["activation"],derivative = True, sample = activations[i-1])


            text = f"Epoch: {epoch}  loss: {self.loss(y_pred = output, y_true = mini_y)}"
            print("\r" + " " * len(text) + "\r", end="", flush=True)
            print(text, end="",flush=True)

            if  epoch - interval  == 0:
            # time.sleep(0.5)
                interval+=interval2
                print()
                y_pred = self.predict(X)
                accuracy = np.mean(np.argmax(y) == np.argmax(y_pred))

                print("\n Accuracy:",accuracy)
                # evaluation = self.evaluate(X, y)
                # print(f"\nEvaluation loss:{evaluation[0]} accuracy:{evaluation[1]}")

            
               
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
        # model_data['loss'] = self.loss
        model_data['loss'] = {
            'name': self.loss.__name__,
            'module': self.loss.__module__

        }
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
        model.loss =getattr(__import__(model_data['loss']['module'], fromlist=[model_data['loss']['name']]), model_data['loss']['name'])
        # loss_module = __import__(model_data['loss']['module'], fromlist=[model_data['loss']['name']])
        # loss_class = getattr(loss_module, model_data['loss']['class'])
        # model.loss = getattr(loss_class, model_data['loss']['name'])
        return model
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss(y_pred, y)
        accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
        return loss, accuracy
