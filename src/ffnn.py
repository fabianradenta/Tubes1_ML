import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
from utils import (
    linear, relu, sigmoid, tanh, softmax, leaky_relu, elu,
    d_linear, d_relu, d_sigmoid, d_tanh, d_softmax,d_leaky_relu, d_elu,
    mse, binary_cross_entropy, categorical_cross_entropy,
    d_mse, d_bce, d_cce, initialize_weights,save_model,load_model,
)

class FFNN :
    def __init__(self, layer_sizes, activation_func, loss_func, weight_init="uniform", learning_rate=0.01, l1_lambda=0, l2_lambda=0):
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
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
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
        elif activation=="leaky_relu" :
            return leaky_relu(x)
        elif activation=="elu" :
            return elu(x)
        else:
            raise ValueError(f"Aktivasi tidak dikenal: {activation}")
        
    def _activation_derivative(self, x, activation) :
        if activation=="linear" :
            return d_linear(x)
        elif activation=="relu" :
            return d_relu(x)
        elif activation=="sigmoid" :
            return d_sigmoid(x)
        elif activation=="tanh" :
            return d_tanh(x)
        elif activation=="softmax" :
            return d_softmax(x)
        elif activation=="leaky_relu" :
            return d_leaky_relu(x)
        elif activation=="elu" :
            return d_elu(x)
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
    
    def compute_regularization_loss(self):
        l1_loss = 0
        l2_loss = 0
        if self.l1_lambda > 0 or self.l2_lambda > 0:
            for layer in self.layers:
                if self.l1_lambda > 0:
                    l1_loss+=np.sum(np.abs(layer["weights"]))
                if self.l2_lambda > 0:
                    l2_loss+=np.sum(np.square(layer["weights"]))
        return self.l1_lambda*l1_loss+0.5*self.l2_lambda*l2_loss
    
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
            if self.l1_lambda > 0:
                l1_grad = np.sign(layer["weights"])
                layer["grad_weights"]+=self.l1_lambda*l1_grad
            if self.l2_lambda > 0:
                l2_grad = layer["weights"]
                layer["grad_weights"]+=self.l2_lambda*l2_grad
            
            layer["grad_bias"] = np.sum(dA, axis=0, keepdims=True)
            error = np.dot(dA, layer["weights"].T)

    def update_weights(self) :
        for layer in self.layers:
            layer["grad_weights"] = np.clip(layer["grad_weights"], -1, 1)
            layer["grad_bias"] = np.clip(layer["grad_bias"], -1, 1)
            layer["weights"] -= self.learning_rate*layer["grad_weights"]
            layer["bias"] -= self.learning_rate*layer["grad_bias"]


    def plot_weight_distribution(self, layers=None):
        """Plot weight distribution for specified layers"""
        if layers is None:
            layers = range(len(self.layers))
        
        plt.figure(figsize=(12, 4*len(layers)))
        for i in layers:
            plt.subplot(len(layers), 1, i+1)
            plt.hist(self.layers[i]['weights'].flatten(), bins=50)
            plt.title(f'Weight Distribution - Layer {i}')
        plt.tight_layout()
        plt.show()
    
    def plot_weight_gradient_distribution(self, layers=None):
        """Plot weight gradient distribution for specified layers"""
        if layers is None:
            layers = range(len(self.layers))
        
        plt.figure(figsize=(12, 4*len(layers)))
        for i in layers:
            plt.subplot(len(layers), 1, i+1)
            if self.layers[i].get('grad_weights') is not None:
                plt.hist(self.layers[i]['grad_weights'].flatten(), bins=50)
                plt.title(f'Weight Gradient Distribution - Layer {i}')
        plt.tight_layout()
        plt.show()

    def visualize_network_structure(self, highlight_weights=True, highlight_gradients=False):
        G = nx.DiGraph()
        pos = {}
        edge_weights = []
        for layer_idx, layer in enumerate(self.layers):
            num_neurons = layer['weights'].shape[1]
            for neuron_idx in range(num_neurons):
                node_name = f'Layer {layer_idx} - Neuron {neuron_idx}'
                G.add_node(node_name)
                pos[node_name] = (layer_idx, neuron_idx - (num_neurons-1)/2)
        
        for layer_idx in range(len(self.layers)-1):
            current_layer_neurons = self.layers[layer_idx]['weights'].shape[1]
            next_layer_neurons = self.layers[layer_idx+1]['weights'].shape[1]
            
            for curr_neuron in range(current_layer_neurons):
                for next_neuron in range(next_layer_neurons):
                    curr_node = f'Layer {layer_idx} - Neuron {curr_neuron}'
                    next_node = f'Layer {layer_idx+1} - Neuron {next_neuron}'
                    
                    if highlight_weights:
                        weight = abs(self.layers[layer_idx]['weights'][curr_neuron, next_neuron])
                    elif highlight_gradients and self.layers[layer_idx].get('grad_weights') is not None:
                        weight = abs(self.layers[layer_idx]['grad_weights'][curr_neuron, next_neuron])
                    else:
                        weight = 1
                    
                    G.add_edge(curr_node, next_node, weight=weight)
                    edge_weights.append(weight)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        if edge_weights:
            norm = colors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            cmap = cm.coolwarm
            
            for (u, v, data) in G.edges(data=True):
                nx.draw_networkx_edges(
                    G, pos, 
                    edgelist=[(u,v)], 
                    edge_color=cmap(norm(data['weight'])),  
                    width=max(0.1, 2 * data['weight']), 
                    alpha=0.6, 
                    arrows=True, 
                    arrowsize=10,
                    ax=ax
                )
            
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  
            
            plt.colorbar(sm, ax=ax, label='Weight/Gradient Magnitude')
        else:
            nx.draw_networkx_edges(
                G, pos, 
                edge_color='blue', 
                width=0.5, 
                alpha=0.6, 
                arrows=True, 
                arrowsize=10,
                ax=ax
            )
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
        
        ax.set_title("Neural Network Structure Visualization")
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def save(self, filename):
        save_model(self, filename)
    
    def load(self, filename):
        load_model(self, filename)
    
    # def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
    #     for epoch in range(epochs):
    #         epoch_train_losses = []
            
    #         for i in range(0, len(X_train), batch_size):
    #             X_batch = X_train[i:i+batch_size]
    #             y_batch = y_train[i:i+batch_size]
                
    #             y_pred = self.forward(X_batch)
    #             self.backward(y_batch, y_pred)
    #             self.update_weights()
                
    #             batch_loss = self._get_loss_function()(y_batch, y_pred)
    #             reg_loss = self.compute_regularization_loss()
    #             total_loss = batch_loss+reg_loss
                
    #             epoch_train_losses.append(total_loss)
            
    #         train_loss = np.mean(epoch_train_losses)
    #         val_loss = self.compute_loss(X_val, y_val)
            
    #         self.history["train_loss"].append(train_loss)
    #         self.history["val_loss"].append(val_loss)
            
    #         if verbose == 1:
    #             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        for epoch in range(epochs):
            epoch_train_losses = []
            
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            if verbose == 1:
                from tqdm import tqdm
                pbar = tqdm(total=len(X_train), desc=f'Epoch {epoch+1}/{epochs}', unit='samples')
            
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights()
                
                batch_loss = self._get_loss_function()(y_batch, y_pred)
                reg_loss = self.compute_regularization_loss()
                epoch_train_losses.append(batch_loss + reg_loss)
                
                if verbose == 1:
                    pbar.update(len(X_batch))
                    pbar.set_postfix({'Train Loss': f'{np.mean(epoch_train_losses):.4f}'})
            
            if verbose == 1:
                pbar.close()
            
            train_loss = np.mean(epoch_train_losses)
            val_loss = self.compute_loss(X_val, y_val)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if verbose == 1:
                print(f'Epoch {epoch+1}/{epochs} - '
                    f'Train Loss: {train_loss:.4f} - '
                    f'Val Loss: {val_loss:.4f}')
        
        return self.history
    

    def compute_loss(self, X, y) :
        y_pred = self.forward(X)
        loss_func = self._get_loss_function()
        data_loss = loss_func(y, y_pred)
        reg_loss = self.compute_regularization_loss()
        return data_loss+reg_loss

    def predict(self, X):
        return self.forward(X)
    
