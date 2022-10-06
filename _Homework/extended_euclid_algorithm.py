def my_euclid(a, b):
	"""Runs the Extended Euclidean Algorithm Recursively"""

	# Check base case and return the gcd
	if b == 0:
		return a, 1, 0

	# Intermediate step to generate the linear combination for z the gcd
	z, x, y = my_euclid(b, a % b)

	# Recalculate the values to return
	gcd, x, y = z, y, x - (a//b)*y
	return gcd, x, y


if __name__ == "__main__":
	print(my_euclid(323, 204))
