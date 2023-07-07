
import numpy as np

class Activations:
    def activator(self, activator_name = "relu", sample = None, derivative = False):
        assert sample is not None
        if activator_name == "relu":
            return self.ReLU(sample, derivative=derivative)
        elif activator_name == "sigmoid":
            return self.sigmoid(sample, derivative=derivative)
        elif activator_name == "softmax":
            return self.softmax(sample, derivative=derivative)
        elif activator_name == "linear":
            return self.linear(sample, deriative=derivative)

    def ReLU(self,x, derivative = False):
        if not derivative:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x, derivative = False):
        if derivative:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1 / (1 + np.exp(-x))
        
    def softmax(self, x, derivative = False):
        if derivative:
            p = self.softmax(x)
            return p * (1 - p)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def linear(self, sample, deriative=False):
        if deriative:
            return 1
        else:
            return sample
    
