# Define the value iteration function for Exercise 16.10
def value_iteration(T, u):
	# Define allowable actions for 1,...,9
	A = {
		1: [2, 4],
		2: [1, 3, 5],
		3: [2, 6],
		4: [1, 5, 7],
		5: [2, 4, 6, 8],
		6: [3, 5, 9],
		7: [4, 8],
		8: [5, 7, 9],
		9: []
	}

	# Use Bellman's Optimality Principle to find each table bottom up
	table_old = [max([u[a] for a in A[s]]) if s != 9 else 0 for s in range(1, 10)]

	# Cycle to find the Tth table
	for t in range(T):
		table_new = [max([u[a] + table_old[a - 1] for a in A[s]]) if s != 9 else 0 for s in range(1, 10)]
		table_old = table_new

	return table_old


if __name__ == "__main__":
	T = 3
	u = {
		1: -1,
		2: 0.7,
		3: -1,
		4: -1,
		5: -1,
		6: -1,
		7: -1,
		8: -1,
		9: -1
	}
	print(value_iteration(T, u))
