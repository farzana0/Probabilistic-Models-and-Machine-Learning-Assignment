import numpy as np
import readmnist


def construct_matrix(samples):
    return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(labels):
    #return labels
    return np.array(labels).reshape(len(labels), 1)


def readdata():
    train_dataset, train_labels = readmnist.trainadata()
    return construct_matrix(train_dataset), construct_labels(train_labels)


def SVM(trainX, trainy, k, iteration, landa):
	w = np.zeros((1, trainX.shape[1]))
	for t in range(iteration):
		At = np.random.choice(trainX.shape[0], size=k)
		X = trainX[At]
		Y = trainy[At]
		Atplus = []
		for i in range(k):
			if Y[i, 0] * np.inner(w, X[i, :]) < 1:
				Atplus.append(i)
		etta = 1.0 / (landa * (t+1))
		Atplusplus = []
		for i in Atplus:
			Atplusplus.append((Y[i] * X[i, :]).reshape(w.shape))
		w = ((1 - etta * landa) * w + etta * (1 / k) * (np.array(Atplusplus).sum(axis=0)))
	return w


def multiclass(num_classes, trainiX, Y):
	w = []
	yi = np.zeros((Y.shape[0], 1))
	for j in range(num_classes):
		print('Class ' + str(j))
		yi[np.where(Y == j)] = 1
		yi[np.where(Y != j)] = -1
		w_i = SVM(trainiX, yi, 600, 3000, (1.0 / 60000.0))
		w.append(np.array(w_i).reshape(trainiX.shape[1], 1))
		yi = np.zeros((trainiX.shape[0], 1))
	return np.array(w)


def accuracy_train(w, trainiX, trainiy):
	limit = trainiX.shape[0]
	counter = 0
	for i in range(limit):
		score = []
		for j in range(10):
			score.append(trainiX[i].dot(w[j]))
		lb_pred = np.argmax(score)
		if lb_pred == trainiy[i]:
			counter += 1
	accuracy = counter / limit
	return accuracy


def accuracy_test(w):
	data_test, label_test = readmnist.testadata()
	test = construct_matrix(data_test)
	x = test
	#x = np.insert(test, 0, 1, axis=1)
	label = construct_labels(label_test)
	limit = test.shape[0]
	counter = 0
	for i in range(limit):
		score = np.zeros((10))
		for j in range(10):
			score[j] = x[i].dot(w[:, j])
		lb_pred = np.argmax(score)
		if lb_pred == label[i]:
			counter += 1

	accuracy = counter / limit
	return accuracy


import numpy as np
import random as rnd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import readmnist


def construct_matrix(samples):
    return np.array(samples, dtype=np.float).reshape(len(samples), 784)


def construct_labels(train_y):
    #return train_y
    return np.array(train_y).reshape(len(train_y), 1)


def readdata():
    train_dataset, train_labels = readmnist.trainadata()
    return construct_matrix(train_dataset), construct_labels(train_labels)


def rbf(x1, x2, sig=1):
	pairwaist_dist = cdist(x1, x2, 'euclidean')
	res = np.exp(-(pairwaist_dist ** 2) / (2 * sig * sig))
	return np.matrix(res)


def nonlinear_svm(train_x, train_y, test_x, test_y, digits):
	dimension = train_x.shape[1]
	limit = len(train_x)
	num_digits = len(digits)

	train_x = train_x.reshape(-1, dimension)
	x = train_x
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)
	target_y = -np.ones((limit, num_digits))

	for i in range(10):
		for k in range(limit):
			if train_y[k] == digits[i]:
				target_y[k, i] = 1

	target_y = np.matrix(target_y)
	epochs = 2
	ld = 2.0 / (2 * limit)
	alphas = []
	rand_idx = np.arange(limit)

	cc = rbf(x, x)

	for k in range(num_digits):

		alpha = np.zeros((limit, 1))
		next_alpha = np.zeros((limit, 1))

		for t in range(epochs):

			# i_t = rnd.randint(0, limit-1)
			rnd.shuffle(rand_idx)
			cnt = 0
			for i_t in rand_idx:

				temp = next_alpha[i_t]
				next_alpha = alpha
				next_alpha[i_t] = temp

				sigma = 0
				for i in range(limit):
					sigma += alpha[i] * target_y[i_t, k] * cc[i_t, i]

				if target_y[i_t, k] * (1 / (ld * (t * limit + cnt + 1))) * sigma < 1.0:
					next_alpha[i_t] = alpha[i_t] + 1
				else:
					next_alpha[i_t] = alpha[i_t]
				alpha = next_alpha
				cnt += 1


		alphas.append(alpha)

	pred_y = np.zeros((num_digits, limit))
	for k in range(num_digits):
		pred_y[k] = (np.multiply(np.tile(target_y[:, k], [limit]), cc).T * alphas[k]).reshape(limit)

	lbls = np.argmax(pred_y, axis=0)
	corr_labels = []
	for i in range(limit):
		if train_y[i, 0] == lbls[i]:
			corr_labels.append(1)
	corr_labels = np.array(corr_labels)
	acc = 1.0 * corr_labels.sum() / limit

	print(acc)


	te_size = len(test_x)
	test_x = np.array(test_x).reshape(-1, dimension)
	t_x = test_x
	t_x = scaler.fit_transform(t_x)

	t_pred_y = np.zeros((num_digits, te_size))
	t_k_cache = rbf(x, t_x)

	for k in range(num_digits):
		t_pred_y[k] = (np.multiply(np.tile(target_y[:, k], [te_size]), t_k_cache).T * alphas[k]).reshape(te_size)

	t_lbls = np.argmax(t_pred_y, axis=0)
	t_corr_labels = []
	for i in range(te_size):
		if t_lbls[i] == test_y[i]:
			t_corr_labels.append(1)
	t_acc = 1.0 * np.array(t_corr_labels).sum() / te_size

	print(t_acc)



if __name__ == "__main__":
	
	train_X, train_y = readdata()
	idx = np.random.randint(train_X.shape[0], size=60000)
	train_X = train_X[idx, :]
	train_y = train_y[idx, :]

	data_test, label_test = readmnist.testadata()
	test = construct_matrix(data_test)
	label_test = construct_labels(label_test)

	nonlinear_svm(train_X, train_y, data_test, label_test, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	train_X, train_y = readdata()
	idx = np.random.randint(train_X.shape[0], size=60000)
	train_X = train_X[idx, :]
	train_y = train_y[idx, :]
	data_test, label_test = readmnist.testadata()
	test = construct_matrix(data_test)
	label_test = construct_labels(label_test)
	print(accuracy_train(multiclass(10, test, label_test), train_X, train_y))
