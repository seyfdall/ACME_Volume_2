# Problem 1.26
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np

# Function that creates a matrix
def make_matrix(N):
	a = np.random.rand(2**N, 2**N)
	return a


def make_vector(N):
	a = np.random.rand(2**N, 1)
	return a


def time_computation(A, B, x, times_1, times_2):
	start = time.time()
	C = np.dot(A, B)
	np.dot(C, x)
	total_time_1 = time.time() - start
	print("(AB)x time: ", str(total_time_1))
	times_1.append(total_time_1)

	start = time.time()
	C = np.dot(B, x)
	np.dot(A, C)
	total_time_2 = time.time() - start
	print("A(Bx) time: ", str(total_time_2))
	times_2.append(total_time_2)
	print("Ratio:", str(total_time_1 / (total_time_2 + 0.00000001)))


if __name__ == "__main__":
	times_1 = []
	times_2 = []
	for i in range(1, 14):
		print("Times for: ", str(i))
		A = make_matrix(i)
		B = make_matrix(i)
		x = make_vector(i)
		time_computation(A, B, x, times_1, times_2)

	plt.plot(range(1, 14), times_1, label="(AB)x times")
	plt.plot(range(1, 14), times_2, label="A(Bx) times")
	plt.legend()
	plt.show()