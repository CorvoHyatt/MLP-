import numpy as np
import os
import struct

def load_mnist(path, kind='train'):
    """Leer los datos de MNIST de la ruta 'path'"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels


class NeuralNetMLP:
    def __init__(self, eta=0.01, epochs=50, random_state=None, init_weight=True, weights=None,
                 shuffle=True, f_activate='sigmoid', n_hidden=[5], minibatch_size=1, loss_function='mse'):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.init_weight = init_weight
        self.weights = weights
        self.shuffle = shuffle
        self.f_activate = f_activate
        self.n_hidden = n_hidden
        self.minibatch_size = minibatch_size
        self.loss_function = loss_function
        self.weights_ = np.array([[0.15, 0.25],[0.20, 0.30],[0.40, 0.50],[0.45, 0.55]])
        self.biases_ = np.array([[0.35, 0.35],[0.60, 0.60]])
        

    def _initialize_weights(self, X_train, y_train):
        if self.init_weight:
            if self.weights is not None:
                self.weights_ = self.weights
            else:
                rgen = np.random.RandomState(self.random_state)
                self.n_features_ = X_train.shape[1]
                self.n_output_ = y_train.shape[1]
                
                # Initialize weights
                self.weights_ = [] 
                for i in range(len(self.n_hidden)):
                    if i == 0:
                        # Input layer to the first hidden layer
                        weights_i = rgen.normal(loc=0.0, scale=0.01, size=(self.n_hidden[0], self.n_features_))
                    else:
                        # Hidden layers
                        weights_i = rgen.normal(loc=0.0, scale=0.01, size=(self.n_hidden[i], self.n_hidden[i-1]))
                    
                    self.weights_.append(weights_i)
                
                # Initialize biases
                self.biases_ = [rgen.normal(loc=0.0, scale=0.01, size=(n, 1)) for n in self.n_hidden]

                print("Weights:", self.weights_)
                print("Biases:", self.biases_)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _tanh(self, z):
        return np.tanh(z)

    def _forward(self, X):
        # Lista para almacenar las activaciones en cada capa
        activations = []
    
        # Capa de entrada
        activation = X
        activations.append(activation)

        # Capas ocultas
        for i in range(len(self.n_hidden)):
            net_input = np.dot(activation, self.weights_[i].T) + self.biases_[i]
            activation = self._sigmoid(net_input) if self.f_activate == 'sigmoid' else self._tanh(net_input)
            activations.append(activation)

        # Capa de salida
        net_input = np.dot(activation, self.weights_[-1].T)
        activation = self._sigmoid(net_input) if self.f_activate == 'sigmoid' else self._tanh(net_input)
        activations.append(activation)

        return activations


    def _compute_loss(self, y_enc, output):
        if self.loss_function == 'bce':
            term1 = -y_enc * (np.log(output))
            term2 = (1 - y_enc) * np.log(1 - output)
            loss = np.sum(term1 - term2)
        else:
            loss = 0.5 * np.sum((y_enc - output) ** 2)
        return loss

    def _get_gradient(self, activations, y_enc):
        gradients = []

        output_error = activations[-1] - y_enc

        # Calcular la derivada de la función de activación en la capa de salida
        if self.f_activate == 'sigmoid':
            activation_derivative = activations[-1] * (1 - activations[-1])
        else:
            activation_derivative = 1.0 - activations[-1]**2

        # Calcular el delta en la capa de salida y el gradiente de los pesos de salida
        delta_output = output_error * activation_derivative
        gradient_output = np.dot(activations[-2].T, delta_output)
        gradients.append(gradient_output)

        # Retropropagación a través de las capas ocultas
        for i in range(len(self.weights_) - 2, -1, -1):
            # Calcular la derivada de la función de activación en la capa oculta
            if self.f_activate == 'sigmoid':
                activation_derivative = activations[i + 1][:, 1:] * (1 - activations[i + 1][:, 1:])
            else:
                activation_derivative = 1.0 - activations[i + 1][:, 1:]**2

            # Calcular el delta en la capa oculta y el gradiente de los pesos de la capa oculta
            delta_hidden = np.dot(delta_output, self.weights_[i + 1][:, 1:]) * activation_derivative
            gradient_hidden = np.dot(activations[i].T, delta_hidden)
            gradients.insert(0, gradient_hidden)

            # Actualizar el delta para la próxima iteración
            delta_output = delta_hidden

        return gradients


    def fit(self, X_train, y_train, X_val, y_val):
        self.n_features_ = X_train.shape[1]
        self.n_output_ = y_train.shape[1]
        self.cost_ = []

        X_data, y_data = X_train.copy(), y_train.copy()
        if not self.init_weight:
            self._initialize_weights()

        for epoch in range(self.epochs):
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatch_size)
            for idx in mini:
                # feedforward
                activations = self._forward(X_data[idx])

                # compute loss
                y_enc = y_data[idx]
                output = activations[-1]
                cost = self._compute_loss(y_enc, output)
                self.cost_.append(cost)

                # compute gradients via backpropagation
                gradients = self._get_gradient(activations, y_enc)

                # update weights
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.eta * gradients[i]

            # compute validation loss
            if X_val is not None and y_val is not None:
                val_activations = self._forward(X_val)
                val_output = val_activations[-1]
                val_loss = self._compute_loss(y_val, val_output)
                print(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {cost:.4f}, Validation Loss: {val_loss:.4f}')

        return self

