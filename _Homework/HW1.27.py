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
	b = np.random.randint(1, 10, size=(2**N))
	return a


def make_identity(N):
	a = np.eye(2**N)
	return a


def time_computation(u, v, x, I, times_1, times_2):
	start = time.time()
	C = I + np.dot(u, np.transpose(v))
	a = np.dot(C, x)
	total_time_1 = time.time() - start
	print("(I + uv^t)x time: ", str(total_time_1))
	times_1.append(total_time_1)

	start = time.time()
	C = np.dot(u, np.dot(np.transpose(v), x))
	b = x + C
	total_time_2 = time.time() - start
	print("x + uv^tx time: ", str(total_time_2))
	times_2.append(total_time_2)
	print("Ratio:", str(total_time_1 / (total_time_2 + 0.00000001)))


if __name__ == "__main__":
	times_1 = []
	times_2 = []
	for i in range(1, 14):
		print("Times for: ", str(i))
		u = make_matrix(i)
		v = make_matrix(i)
		x = make_vector(i)
		I = make_identity(i)
		time_computation(u, v, x, I, times_1, times_2)

	plt.plot(range(1, 14), times_1, label="(I + uv^t)x times")
	plt.plot(range(1, 14), times_2, label="x + uv^tx times")
	plt.legend()
	plt.show()