import numpy as np
import readmnist

def construct_matrix(samples):
	return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(labels):
	#return labels
	return np.array(labels).reshape(len(labels))


def regression(Y, X):
	w = np.zeros((785, 10))
	yi = np.zeros((60000))
	mt = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), (X.T))
	print(mt.shape)
	for j in range(10):
		yi[np.where(Y == j)] = 1
		print(yi.sum())
		w[:,j] = np.matmul(mt, yi)
		#np.random.shuffle(w[:, j])
		print(w[:,j].sum())
		yi = np.zeros((60000))
	return w


def accuracy_train(w):
	data_test, label_test = readmnist.testadata()
	test = construct_matrix(data_test)
	x = np.insert(test, 0, 1, axis=1)
	label = construct_labels(label_test)
	limit = test.shape[0]

	counter = 0
	for i in range(limit):
		score = np.zeros((10))
		for j in range(10):
			score[j] = np.matmul(x[i], w[:,j])
		lb_pred = np.argmax(score)
		if lb_pred == label[i]:
			counter += 1

	accuracy = counter / limit
	return accuracy




np.random.seed(12)

def construct_matrix(samples):
	return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(labels):
	#return labels
	return np.array(labels, dtype=float).reshape(len(labels))

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 10000 == 0:
            #print(log_likelihood(features, target, weights))
            pass
        
    return weights






X , Y = readmnist.trainadata()
X = construct_matrix(X)
Y = construct_labels(Y)


def multiclass_logistic_regression(Y, X):
	w = np.zeros((785, 10))
	yi = np.zeros((60000))
	for j in range(10):
		yi[np.where(Y == j)] = 1
		w[:,j] = logistic_regression(X, yi, num_steps = 3000, learning_rate = 5e-5, add_intercept=True )
		yi = np.zeros((60000))
	return w

def accuracy_train(w):
	data_test, label_test = readmnist.testadata()
	test = construct_matrix(data_test)
	x = np.insert(test, 0, 1, axis=1)
	label = construct_labels(label_test)
	limit = test.shape[0]

	counter = 0
	for i in range(limit):
		score = np.zeros((10))
		for j in range(10):
			score[j] = np.matmul(x[i], w[:,j])
		lb_pred = np.argmax(score)
		if lb_pred == label[i]:
			counter += 1

	accuracy = counter / limit
	return accuracy


if __name__ == "__main__":

	X , Y = readmnist.trainadata()
	X = construct_matrix(X)
	Y = construct_labels(Y)
	print(type(Y))
	X = np.insert(X, 0, 1, axis=1)
	w = np.zeros((785, 10))
	yi = np.zeros((60000))
	mt = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), (X.T))
	print(X[100].sum())

	print(mt.shape)
	for j in range(10):
		yi[np.where(Y == j)] = 1
		w[:,j] = np.matmul(mt, yi)
		#np.random.shuffle(w[:, j])
		yi = np.zeros((60000))
	print(accuracy_train(w))
	X , Y = readmnist.trainadata()
	X = construct_matrix(X)
	Y = construct_labels(Y)
	print(accuracy_train(multiclass_logistic_regression(Y, X)))

