import numpy as np
from utils import (
    linear, relu, sigmoid, tanh, softmax, 
    d_linear, d_relu, d_sigmoid, d_tanh,
    mse, binary_cross_entropy, categorical_cross_entropy,
    d_mse, d_bce, d_cce, initialize_weights
)

class FFNN :
    def __init__(self, layer_sizes, activation_func, loss_func, weight_init="uniform", learning_rate=0.01):
        self.layers = []
        for i in range (len(layer_sizes)-1) :
            layer = {
                "weights" : initialize_weights(layer_sizes[i],layer_sizes[i+1], weight_init),
                "bias" : np.zeros((1, layer_sizes[i+1])),
                "activation" : activation_func[i],
                "grad_weight" : None,
                "grad_bias" : None
            }
            self.layers.append(layer)
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.history = {"train_loss" : [], "val_loss" : []}

    def _apply_activation(self, x, activation) :
        if activation=="linear" :
            return linear(x)
        elif activation=="relu" :
            return relu(x)
        elif activation=="sigmoid" :
            return sigmoid(x)
        elif activation=="tanh" :
            return tanh(x)
        elif activation=="softmax" :
            return softmax(x)
        
    def _activation_derivative(self, x, activation) :
        if activation=="linear" :
            return d_linear(x)
        elif activation=="relu" :
            return d_relu(x)
        elif activation=="sigmoid" :
            return d_sigmoid(x)
        elif activation=="tanh" :
            return d_tanh(x)
        elif activation=='softmax' :
            return np.ones_like(x)
        else:
            raise ValueError(f"Aktivasi tidak dikenal: {activation}")
        
    def _get_loss_function(self) :
        if self.loss_func=="mse" :
            return mse
        elif self.loss_func=="bce" :
            return binary_cross_entropy
        elif self.loss_func=="cce" :
            return categorical_cross_entropy
        
    def _get_loss_derivative(self) :
        if self.loss_func=="mse" :
            return d_mse
        elif self.loss_func=="bce" :
            return d_bce
        elif self.loss_func=="cce" :
            return d_cce
        
    def forward(self, x) :
        self.input = x
        for layer in self.layers :
            layer["input"] = x
            Z = np.dot(x, layer["weights"]) + layer["bias"]
            A = self._apply_activation(Z, layer["activation"])
            layer["output"] = A
            x = A
        return x
    
    def backward(self, y_true, y_pred) :
        if self.layers[-1]["activation"] == "softmax" and self.loss_func == "cce":
            error = y_pred-y_true
        else:
            loss_derivative_func = self._get_loss_derivative()
            error = loss_derivative_func(y_true, y_pred)
        for i in reversed(range(len(self.layers))) :
            layer = self.layers[i]
            A = layer["output"]
            dA = error*self._activation_derivative(A, layer["activation"])
            
            if i==0:
                prev_output = self.input
            else:
                prev_output = self.layers[i-1]["output"]
            
            layer["grad_weights"] = np.dot(prev_output.T, dA)
            layer["grad_bias"] = np.sum(dA, axis=0, keepdims=True)
            error = np.dot(dA, layer["weights"].T)

    def update_weights(self) :
        for layer in self.layers:
            layer["grad_weights"] = np.clip(layer["grad_weights"], -1, 1)
            layer["grad_bias"] = np.clip(layer["grad_bias"], -1, 1)
            layer["weights"] -= self.learning_rate*layer["grad_weights"]
            layer["bias"] -= self.learning_rate*layer["grad_bias"]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1) :
        for epoch in range(epochs) :
            for i in range(0, len(X_train), batch_size) :
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights()
            
            train_loss = self.compute_loss(X_train, y_train)
            val_loss = self.compute_loss(X_val, y_val)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def compute_loss(self, X, y) :
        y_pred = self.forward(X)
        loss_func = self._get_loss_function()
        return loss_func(y,y_pred)

    def predict(self, X):
        return self.forward(X)