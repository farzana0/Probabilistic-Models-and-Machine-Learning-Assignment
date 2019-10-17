
import numpy as np
import readmnist 
from scipy.special import expit

def construct_matrix(samples):
    return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(labels):
    #return labels
    return np.array(labels).reshape(len(labels), 1)


def readdata():
    train_dataset, train_labels = readmnist.trainadata()
    return construct_matrix(train_dataset), construct_labels(train_labels)


def normalizedata(data):
    num_features = data.shape[1]

    mean = np.array([data[:,j].mean() for j in range(num_features)]).reshape(num_features)
    std = np.array([data[:,j].std() for j in range(num_features)]).reshape(num_features)

    for i in range(num_features):
        if float(std[i]) != 0:
            data[:, i] = (data[:, i] - float(mean[i])) * (1 / float(std[i]))
        else:
            data[:, i] = np.ones((data.shape[0]))
    return data



class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.l2_m = 0
        self.l1_m = 0
        self.l2_v = 0
        self.l1_v = 0
        self.t = 0
        
        # Weights Initilization
        self.w0 = np.random.normal(0.0, 0.1, (self.input_nodes, self.hidden_nodes))
        self.w1 = np.random.normal(0.0, 0.1, (self.hidden_nodes, self.output_nodes))

        def sigmoid(x, deriv=False):
            
            if deriv:
                return x*(1-x)
            return expit(x)
        
        self.activation_function = sigmoid
        
        def softmax(X, deriv = False):
            if deriv:
                J = - X[..., None] * X[:, None, :] # off-diagonal Jacobian
                iy, ix = np.diag_indices_from(J[0])
                J[:, iy, ix] = X * (1. - X) # diagonal
                return -J.sum(axis=1) # sum across-rows for each sample

            exps = np.exp(X - np.max(X))
            return exps / np.sum(exps)


        self.softmax = softmax

        def delta_cross_entropy(X, y):
            m = y.shape[0]
            grad = (X)
            grad[range(m),y] -= 1
            grad = grad/m
            return grad

        self.delta_cross_entropy = delta_cross_entropy

    def train(self, features, targets, optimizer, decay_rate_1 = None, 
              decay_rate_2 = None, epsilon = None):

        batch_size = targets.shape[0]
        dw0 = np.zeros(self.w0.shape)
        dw1 = np.zeros(self.w1.shape)
        for k in range(batch_size):
            # Feed Forward
            l0 = features[k].reshape(1,-1)
            l1 = self.activation_function(l0.dot(self.w0))
            l2 = self.softmax(l1.dot(self.w1))

            prime = np.zeros((1, l2.shape[1]))
            prime[0, targets[k]] = 1

            # Backpropagation       
            l2_error = (l2 - prime)
            l2_delta = l1.T.dot(l2_error)
            dw1 += l2_delta
            l2_delta = dw1
            l1_error = l2_error.dot(self.w1.T) * self.activation_function(l1, deriv=True)
            l1_delta = l0.T.dot(l1_error)
            dw0 += l1_delta
            l1_delta = dw0

        if optimizer == 'sgd':
            self.w1 -= self.lr * dw1 / targets.shape[0]
            self.w0 -= self.lr * dw0 / targets.shape[0]
            
        if optimizer == 'adam':
            # Gradients for each layer
           g1 = l1.dot(dw1)
           g0 = l0.dot(dw0)

           self.t += 1 # Increment Time Step
           # Computing 1st and 2nd moment for each layer
           self.l2_m = self.l2_m * decay_rate_1 + (1- decay_rate_1) * g1
           self.l1_m = self.l1_m * decay_rate_1 + (1- decay_rate_1) * g0
           self.l2_v = self.l2_v * decay_rate_2 + (1- decay_rate_2) * (g1 ** 2)
           self.l1_v = self.l1_v * decay_rate_2 + (1- decay_rate_2) * (g0 ** 2)
           l2_m_corrected = self.l2_m / (1-(decay_rate_1 ** self.t))
           l2_v_corrected = self.l2_v / (1-(decay_rate_2 ** self.t))
           # Computing bias-corrected moment
           l1_m_corrected = self.l1_m / (1-(decay_rate_1 ** self.t))
           l1_v_corrected = self.l1_v / (1-(decay_rate_2 ** self.t))
           # Update Weights
           w1_update = l2_m_corrected / (np.sqrt(l2_v_corrected) + epsilon)
           w0_update = l1_m_corrected / (np.sqrt(l1_v_corrected) + epsilon)

           self.w1 -= (self.lr * w1_update)
           self.w0 -= (self.lr * w0_update)
            
    def run(self, features):
        l0 = features
        l1 = self.activation_function(np.dot(l0, self.w0))
        l2 = self.softmax(np.dot(l1, self.w1))
        return l2


def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def MSE(y, Y):
    return np.mean((y-Y)**2)


def cross_entropy(Y, y):
    m = y.shape[0]
    # p = softmax(Y)
    p = Y
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


import time
def test_accuracysgd(idx):
    test_X, test_y = readmnist.testadata()
    test_X = construct_matrix(test_X)
    test_y = construct_labels(test_y)
    # test_X = normalizedata(test_X)
    
    def test_model(network):
        test_predictions = network.run(test_X)
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            total += 1
            if np.argmax(np.array(test_predictions[i])) == test_y[i]:
                correct += 1
        return correct/total
    print(test_model(network_sgd))


def test_accuracy(idx):
    test_X, test_y = readdata()

    #test_X = normalizedata(test_X)
    test_X = test_X[idx, :]
    test_y = test_y[idx, :]
    def test_model(network):
        test_predictions = network.run(test_X)
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            total += 1
            if np.argmax(np.array(test_predictions[i])) == test_y[i]:
                correct += 1
        return correct/total
    print(test_model(network_adam))

def build_network(network, epochs, optimizer, batch_size = None):
    losses = {'train':[], 'validation':[]} # For Plotting of MSE
    start = time.time()
    
        
    # Iterating Over Epochs
    for i in range(epochs):
        
        if optimizer == 'sgd':
            # Iterating over mini batches
            num_batches = int(train_X.shape[0] / batch_size)
            for k in range(num_batches):
                batch = np.random.choice(train_X.shape[0], size=batch_size)
                X, y = train_X[batch], train_y[batch]
                network.train(X, y, optimizer)

                train_y_one = np.zeros((train_X.shape[0], 10))
                for j in range(train_X.shape[0]):
                    train_y_one[j, train_y[j]] = 1

                #train_loss = MSE(network.run(train_X), train_y_one)
                #val_loss = MSE(network.run(val_X), val_y)
                #train_loss = cross_entropy(network.run(train_X), train_y)
                # val_loss = cross_entropy(network.run(val_X), val_y)

            if i % 1 == 0:
                #print('Epoch {}, Train Loss: {}, Val Loss: {}'.format(i, train_loss, val_loss))
                print('Epoch ' + str(i))
                test_accuracysgd(idx)

        if optimizer == 'adam':
            network.train(train_X, 
                          train_y, 
                          optimizer,
                          decay_rate_1 = 0.9,
                          decay_rate_2 = 0.99,
                          epsilon = 10e-8)

            #train_loss = MSE(network.run(train_X), train_y)
            #val_loss = MSE(network.run(val_X), val_y)
            #train_loss = cross_entropy(network.run(train_X), train_y)
            #val_loss = cross_entropy(network.run(val_X), val_y)

            if i % 1 == 0:
                #print('Epoch {}, Train Loss: {}, Val Loss: {}'.format(i, train_loss, val_loss))
                print('Epoch ' + str(i))
                test_accuracy(idx)

        #losses['train'].append(train_loss)
        #losses['validation'].append(val_loss)
        
    print('Time Taken:{0:.4f}s'.format(time.time()-start))
    return losses


epochs = 2000
learning_rate = 0.01
hidden_nodes = 800
output_nodes = 10
batch_size = 600
'''
zero_labels, zero_vecs = readmnist.getlabel2(0)

labels_0, vecs_0 = readmnist.getlabel2(0)
matrix_f = construct_matrix(vecs_0)
labels_f = construct_labels(labels_0)

for i in range(1, 4):
    labels_ff, matrix_ff = readmnist.getlabel2(i)
    X = np.vstack([matrix_f, construct_matrix(matrix_ff)])
    Y = np.vstack([labels_f, construct_labels(labels_ff)])
    matrix_f = X
    labels_f = Y

X = np.insert(X, 0, 1, axis=1)
'''

train_X, train_y = readdata()
#train_X = normalizedata(train_X)
idx = np.random.randint(train_X.shape[0], size=6000)
train_X = train_X[idx, :]
train_y = train_y[idx, :]

test_X, test_y = readdata()
# test_X = normalizedata(test_X)

val_X, val_y = train_X, train_y

network_adam = NeuralNetwork(train_X.shape[1], hidden_nodes, output_nodes, learning_rate)
#network_sgd = NeuralNetwork(train_X.shape[1], hidden_nodes, output_nodes, learning_rate)

# print('Training Model with Adam')
losses_adam = build_network(network_adam, epochs, 'adam')

print('Training Model with SGD')
#losses_sgd = build_network(network_sgd, epochs, 'sgd', batch_size)


def test_accuracy(idx):
    test_X, test_y = readdata()

    # test_X = normalizedata(test_X)
    test_X = test_X[idx, :]
    test_y = test_y[idx, :]
    def test_model(network):
        test_predictions = network.run(test_X)
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            total += 1
            if np.argmax(np.array(test_predictions[i])) == test_y[i]:
                correct += 1
        return correct/total
    print(test_model(network_adam))

def test_accuracysgd(idx):
    test_X, test_y = readmnist.testadata()
    test_X = construct_matrix(test_X)
    test_y = construct_labels(test_y)
    # test_X = normalizedata(test_X)
    
    def test_model(network):
        test_predictions = network.run(test_X)
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            total += 1
            if np.argmax(np.array(test_predictions[i])) == test_y[i]:
                correct += 1
        return correct/total
    print(test_model(network_sgd))

#test_accuracy(idx)
test_accuracysgd(idx)