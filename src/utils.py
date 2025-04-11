import numpy as np
import matplotlib.pyplot as plt

# ----------- ACTIVATION FUNCTIONS -----------
def linear(x) :
    return x

def relu(x) :
    return np.maximum(0,x)

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def tanh(x) :
    return np.tanh(x) 

def softmax(x) : #vektor dalam array
    x = x-np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.where(x>0, x, alpha*x)

def elu(x, alpha=1.0):
    return np.where(x>0, x, alpha*(np.exp(x)-1))


# ----------- TURUNAN ACTIVATION FUNCTIONS -----------
def d_linear(x) :
    return np.ones_like(x)

def d_relu(x) :
    return (x>0).astype(float)

def d_sigmoid(x) :
    s = sigmoid(x)
    return s*(1-s)

def d_tanh(x) :
    return 1-np.tanh(x)**2

def d_softmax(x) :
    return np.ones_like(x)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x>0, 1, alpha)

def d_elu(x, alpha=1.0):
    return np.where(x>0, 1, alpha * np.exp(x))


# ----------- LOSS FUNCTIONS -----------
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))


# ----------- TURUNAN LOSS FUNCTIONS -----------
def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def d_bce(y_true, y_pred):
    return (y_pred-y_true)/(y_pred*(1-y_pred)+1e-9)

def d_cce(y_true, y_pred):
    return y_pred-y_true

# ----------- INITIALIZE WEIGHT -----------
def initialize_weights(input_size, output_size, method="uniform", **kwargs):
    if method == "zero":
        return np.zeros((input_size, output_size))
    elif method == "uniform":
        low = kwargs.get("low", -0.1)
        high = kwargs.get("high", 0.1)
        return np.random.uniform(low, high, (input_size, output_size))
    elif method == "normal":
        mean = kwargs.get("mean", 0)
        std = kwargs.get("std", 0.01)
        return np.random.normal(mean, std, (input_size, output_size))
    elif method == "xavier":
        limit = np.sqrt(6/(input_size+output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))
    elif method=="he" :
        std = np.sqrt(2/input_size)
        return np.random.normal(0, std, (input_size, output_size))
    

# ----------- PLOTTING LOST -----------
def plot_loss_history(history):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()