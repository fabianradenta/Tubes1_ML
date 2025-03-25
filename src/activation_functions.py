import numpy as np

def Linear(x) :
    return x

def ReLU(x) :
    return np.maximum(0,x)

def Sigmoid(x) :
    return 1/(1+np.exp(-x))

def tanh(x) :
    return np.tanh(x) 

def softmax(x) : #vektor dalam array
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x) 

# def test_activation_functions():
#     # Data uji: skalar dan array
#     scalar = 2.0
#     array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

#     print("Testing Linear function...")
#     print(f"Linear({scalar}) =", Linear(scalar))
#     print(f"Linear({array}) =", Linear(array))

#     print("\nTesting ReLU function...")
#     print(f"ReLU({scalar}) =", ReLU(scalar))
#     print(f"ReLU({array}) =", ReLU(array))

#     print("\nTesting Sigmoid function...")
#     print(f"Sigmoid({scalar}) =", Sigmoid(scalar))
#     print(f"Sigmoid({array}) =", Sigmoid(array))

#     print("\nTesting tanh function...")
#     print(f"tanh({scalar}) =", tanh(scalar))
#     print(f"tanh({array}) =", tanh(array))

#     print("\nTesting Softmax function...")
#     vector = np.array([2.0, 1.0, 0.1])
#     print(f"softmax({vector}) =", softmax(vector))

# # Panggil fungsi pengujian
# test_activation_functions()