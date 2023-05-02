import numpy as np
import matplotlib.pyplot as plt


# problem 9.9a
def bary_weights(points):
	" takes points (array) and returns barycentric weights (array)"
	weights = np.ones_like(points)

	# Using the weights generation method
	for j in range(len(weights)):
		for i in range(len(weights)):
			if i != j:
				weights[j] *= (points[j] - points[i])
		weights[j] = 1.0 / weights[j]
	return weights


# problem 9.9b
def eval_poly(x, y, x0):
	" takes in x (array), y (array), x0 (float), returns interpolation at x (float)"
	# If the value already exists in the array-
	if x0 in x:
		return y[list(x).index(x0)]

	# Find weights
	weights = bary_weights(x)

	# Calculate the fraction
	weight_frac = weights / (x0 - x)

	# Calculate the value
	val = np.sum(weight_frac * y) / np.sum(weight_frac)
	return val


# problem 9.10
def prob_9_10():
	sample_domain = np.linspace(-1,1,100)
	for n in range(2,21):
		# Compute abs(x)
		domain = np.linspace(-1,1,n+1)
		y_range = np.abs(domain)
		y_range_comp = np.abs(sample_domain)

		# Compute the interpolation of abs(x)
		sample = np.array([eval_poly(domain,y_range,x) for x in sample_domain])

		# Plot both
		plt.subplot(4, 5, n - 1)
		plt.plot(sample_domain, y_range_comp)
		plt.plot(sample_domain, sample)
		plt.title(f"n={n}")
		print(f"n={n}:{np.max(np.abs(sample - y_range_comp))}")
	plt.tight_layout()
	plt.show()

# your plots
# print smallest error polynomial
print(eval_poly(np.linspace(-1,1,10), np.abs(np.linspace(-1,1,10)), 0))
prob_9_10()
print('n:\t', '9')
print('error:\t', '0.06497')