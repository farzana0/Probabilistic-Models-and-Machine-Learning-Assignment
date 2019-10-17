import numpy as np
import readmnist
from matplotlib import pyplot as plt




def pca(all_samples, color):

	def construct_matrix(samples):
		leng = len(samples)
		matrixfour = np.array(samples).reshape(leng, 784)
		return matrixfour

	data = construct_matrix(all_samples)
	list = []
	for i in range(784):
		list.append(np.array(data[:, i]))
	cov_mat = np.cov(list).real

	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i].real) for i in range(len(eig_val_cov))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	eig_vals =[i[0] for i in eig_pairs]
	matrix_w = eig_pairs[0][1].reshape(784, 1)

	def distortion(dim, eig_vals):
		return sum(eig_vals[dim:])
	distortions = [distortion(2, eig_vals), distortion(10, eig_vals), distortion(50, eig_vals), distortion(100, eig_vals), distortion(200, eig_vals), distortion(300, eig_vals), distortion(500, eig_vals)]
	x = [2, 10, 50, 100, 200, 300, 500]
	
	plt.plot(x, distortions)
	plt.title('Total distortion errors as a function of PCA dimensions')
	plt.xlabel('PCA dimensions')
	plt.ylabel('Total distortion error')
	plt.show()
	plt.savefig('distortionPCA')

	for i in range(1, 100):
		matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(784, 1)))
	transformed = np.matmul(data, matrix_w)
	#plt.plot(transformed)
	plt.plot(transformed[0:5842, 0], transformed[0:5842, 1], 'o', markersize=3, color= color, alpha=0.5, label='Digit \'4\'')
	plt.plot(transformed[5842:6265+5842, 0], transformed[5842:6265+5842, 1], '^', markersize=3, color='red', alpha=0.5, label='Digit \'7\'')
	plt.plot(transformed[6265+5842:, 0], transformed[6265+5842:, 1], '*', markersize=3, color='green', alpha=0.5, label='Digit \'8\'')

	#plt.xlim([-2000,2000])
	#plt.ylim([-2000,2000])
	#plt.xlabel('x_values')
	#plt.ylabel('y_values')
	plt.legend()
	#plt.title('Transformed samples with class labels')
	plt.savefig('PCA.png')
	plt.show()




def LDA(all_samples, labels):
	prev = 0
	mean_vectors = []
	for cl in range(1, 4):
		indexes = [i for i, j in enumerate(labels) if j == cl]
		indexes = max(indexes)
		mean_vectors.append(np.mean(all_samples[prev:indexes,:], axis=0))
		prev =indexes
		print('Mean Vector class %s: %s\n' % (cl, len(mean_vectors[cl - 1])))
	prev = 0
	S_W = np.zeros((784, 784))
	for cl, mv in zip(range(1, 4), mean_vectors):
		class_sc_mat = np.zeros((784, 784))  # scatter matrix for every class
		indexes = [i for i, j in enumerate(labels) if j == cl]
		indexes = max(indexes)
		for row in all_samples[prev:indexes, :]:
			prev = indexes
			row, mv = row.reshape(784, 1), mv.reshape(784, 1)  # make column vectors
			class_sc_mat += (row - mv).dot((row - mv).T)
		S_W += class_sc_mat  # sum class scatter matrices
	print('within-class Scatter Matrix:\n', S_W)
	overall_mean = np.mean(all_samples, axis=0)

	S_B = np.zeros((784, 784))
	for i, mean_vec in enumerate(mean_vectors):
		indexes = [ii for ii, j in enumerate(labels) if j == i+1]
		n = len(indexes)
		#n = all_samples[len(indexes), :].shape[0]
		mean_vec = mean_vec.reshape(784, 1)  # make column vector
		overall_mean = overall_mean.reshape(784, 1)  # make column vector
		S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

	print('between-class Scatter Matrix:\n', S_B)
	print(S_B.shape)
	print(np.linalg.pinv(S_W).shape)
	eig_vals, eig_vecs = np.linalg.eig(np.matmul(np.linalg.pinv(S_W), S_B))

	for i in range(len(eig_vals)):
		eigvec_sc = eig_vecs[:, i].reshape(784, 1)
		print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
		print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

	# Visually confirm that the list is correctly sorted by decreasing eigenvalues

	W = np.hstack((eig_pairs[0][1].reshape(784, 1), eig_pairs[1][1].reshape(784, 1)))
	X_lda = all_samples.dot(W)

	def plot_step_lda():
		label_dict = {1: 'Digit: 4', 2: 'Digit: 7', 3: 'Digit: 8'}
		ax = plt.subplot(111)
		prev = 0
		for label, marker, color in zip(range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
			indexes = [i for i, j in enumerate(labels) if j == label]
			indexes = max(indexes)
			plt.scatter(x=X_lda[:, 0].real[prev:indexes],
						y=X_lda[:, 1].real[prev:indexes],
						marker=marker,
						color=color,
						alpha=0.5,
						label=label_dict[label]
						)
			prev = indexes

		

		leg = plt.legend(loc='upper right', fancybox=True)
		leg.get_frame().set_alpha(0.5)
		plt.title('LDA: MNIST \'4\', \'7\' and \'8\' projection onto the first 2 linear discriminants')

		plt.show()
		plt.savefig('LDA')

	plot_step_lda()


def scatter(x, colors):
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
   

    # We add the labels for each digit.
    txts = []
    for i in [4, 7, 8]:
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


if __name__ == "__main__":

	sample_four = readmnist.getlabel(4)
	sample_seven = readmnist.getlabel(7)
	sample_eight = readmnist.getlabel(8)
	all_samples = []
	all_samples = sample_four
	for i in sample_seven:
		all_samples.append(i)
	for i in sample_eight:
		all_samples.append(i)
	pca(all_samples, color = 'blue')
	sample_four = readmnist.getlabel(4)
	sample_seven = readmnist.getlabel(7)
	sample_eight = readmnist.getlabel(8)
	all_samples = []
	all_samples = sample_four
	for i in sample_seven:
		all_samples.append(i)
	for i in sample_eight:
		all_samples.append(i)
	labels = []
	for i in range(5842):
		labels.append(1)
	for i in range(6265):
		labels.append(2)
	for i in range(5851):
		labels.append(3)
	def construct_matrix(samples):
		leng = len(samples)
		matrixfour = np.array(samples).reshape(leng, 784)
		return matrixfour
	all_samples = construct_matrix(all_samples)
	LDA(all_samples, labels)
	digits_proj = TSNE(random_state=RS).fit_transform(X)
	scatter(digits_proj, y[0])





