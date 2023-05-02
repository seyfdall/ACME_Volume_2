from matplotlib import pyplot as plt
import numpy as np
import pytest


# Function to graph cancer rates
def cancer_graph():
	# Setup initial values
	false_pos = 0.05
	false_neg = 0.05
	incidence = 0.004

	# Setup initial data structures
	domain_1 = np.linspace(0.001, 0.1, 10000)
	rates_1 = [0 for i in range(10000)]
	domain_2 = np.linspace(0.001, 0.1, 10000)
	rates_2 = [0 for i in range(10000)]
	domain_3 = np.linspace(0.001, 0.05, 10000)
	rates_3 = [0 for i in range(10000)]

	# False Positive Rate Ranges
	j = 0
	for i in domain_1:
		num = (1 - false_neg) * incidence
		den = (1 - false_neg) * incidence + i * (1 - incidence)
		rates_1[j] = num / den
		j += 1

	plt.plot(domain_1, rates_1)
	plt.title("False Positive Rate Graph")
	plt.xlabel("False Positive Rates")
	plt.ylabel("Probabilities")
	plt.show()

	# False Negative Rate Ranges
	j = 0
	for f_neg in domain_2:
		num = (1 - f_neg) * incidence
		den = (1 - f_neg) * incidence + false_pos * (1 - incidence)
		rate = num / den
		rates_2[j] = rate
		j += 1

	plt.plot(domain_2, rates_2)
	plt.title("False Negative Rate Graph")
	plt.xlabel("False Negative Rates")
	plt.ylabel("Probabilities")
	plt.show()

	# Incidence Rate Ranges
	j = 0
	for i in domain_3:
		num = (1 - false_neg) * i
		den = (1 - false_neg) * i + false_pos * (1 - i)
		rates_3[j] = num / den
		j += 1

	plt.plot(domain_3, rates_3)
	plt.title("Incidence Rate Graph")
	plt.xlabel("Incidence Rates")
	plt.ylabel("Probabilities")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	cancer_graph()