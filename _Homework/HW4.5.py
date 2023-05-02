def knapsack(target, items):
	"""Problem 4.26 - 4.27
	Calculate the maximum value that can be
	carried in the knapsack for the given items
	and target.  Also return a list of items which
	should be included to achieve the max value."""

	# Initialize back pointer array
	totals = [[0 for i in range(len(items))] for j in range(target + 1)]

	# Cycle through every weight up to the target value
	for i in range(1, target + 1):
		max_totals = totals[i]

		# Cycle through every item and see if previous count is better
		for item in items:
			if item[0] <= i:
				prev = totals[i - item[0]].copy()
				for j in range(len(items)):
					if items[j] == item and prev[j] == 0:
						prev[j] += 1
						break
				if dot_product(max_totals, items) < dot_product(prev, items) or sum(max_totals) == 0:
					max_totals = prev
		totals[i] = max_totals

	return dot_product(totals[-1], items), totals[-1]


def dot_product(arr1, arr2):
	"""Dot product bit array with values to return sum"""
	sum = 0
	for i in range(len(arr1)):
		sum += arr1[i] * arr2[i][1]
	return sum


if __name__ == "__main__":
	print(knapsack(100, [(20, 0.5), (100, 1.0)]))