
import numpy as np

class Activations:
    def activator(self, activator_name = "relu", sample = None, deriative = False):
        assert sample is not None
        if deriative:
            if activator_name == "relu":
                return self.ReLU_deriative(sample)
            elif activator_name == "sigmoid":
                return self.sigmoid_deriative(sample)
            elif activator_name == "softmax":
                return self.softmax_derivative(sample)
        if activator_name == "relu":
            return self.ReLU(sample)
        elif activator_name == "sigmoid":
            return self.sigmoid(sample)
        elif activator_name == "softmax":
            return self.softmax(sample)

    def ReLU(self,x):
        return np.maximum(0, x)

    def ReLU_derivative(self,x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def softmax_derivative(self, x):
        p = self.softmax(x)
        return p * (1 - p)
    
