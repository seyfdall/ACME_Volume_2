import time
from matplotlib import pyplot as plt


def naive_fib(n):
	"""Naive fibonnaci program"""
	if n == 0 or n == 1:
		return 1
	return naive_fib(n - 1) + naive_fib(n - 2)


def memoized_fib(n):
	"""Memoized Fibonacci"""

	# Handle base case and setup initial lists
	if n == 0 or n == 1:
		return 1
	L = [0 for i in range(n + 1)]
	L[0] = 1
	L[1] = 1

	def memoized_fib_helper(n, L):
		"""Memoized Fibonacci helper to recurse down using memoization"""
		if L[n] != 0:
			return L[n]
		else:
			L[n] = memoized_fib_helper(n - 1, L) + memoized_fib_helper(n - 2, L)
			return L[n]

	# Call the helper and return
	memoized_fib_helper(n, L)
	return L[-1]


def bottom_up_fib(n):
	"""Bottum up Fibonnaci program"""
	# Initial list
	L = [1, 1]

	# Iterate and append
	for k in range(2, n + 1):
		L.append(L[-1] + L[-2])

	# Return last element
	return L[-1]


def prob_1(n):
	"""Make plots for PDF
	The largest n that worked for all three in under a minute was n = 41
	"""

	# Initial lists for x and y values
	naive_times = [0]
	memoized_times = [0]
	bottom_times = [0]
	domain = [x for x in range(n + 1)]

	for k in range(n):
		# Time the naive_fib
		start = time.time()
		print(naive_fib(k))
		finish = time.time()
		naive = finish - start
		print("Naive Time:", naive)

		# Time the memoized_fib
		start = time.time()
		print(memoized_fib(k))
		finish = time.time()
		memoized = finish - start
		print("Memoized Time:", memoized)

		# Time the bottom_up_fib
		start = time.time()
		print(bottom_up_fib(k))
		finish = time.time()
		bottom = finish - start
		print("Bottom-Up Time:", bottom)

		# Append times
		naive_times.append(naive)
		memoized_times.append(memoized)
		bottom_times.append(bottom)

	# Plot each graph on one plot
	plt.plot(domain, naive_times, 'g-', label="Naive")
	plt.plot(domain, bottom_times, 'r-', label="Bottom-Up")
	plt.plot(domain, memoized_times, 'b-', label="Memoized")
	plt.xlabel("n values")
	plt.ylabel("Times")
	plt.title("Problem 1: Fibonacci")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.show()


def naive_coins(val, coins):
	"""Naive implementation of coin change"""

	# Initialize starting coins and base case
	coins_needed = [0 for i in range(len(coins))]
	if val in coins:
		coins_needed[coins.index(val)] += 1
		return 1, coins_needed
	else:
		# Naively recursively iterate
		ret_val = [-1, coins_needed]
		for coin in coins:
			if coin <= val:
				temp_val = naive_coins(val - coin, coins)
				if ret_val[0] == -1 or temp_val[0] + 1 < ret_val[0]:
					ret_val[0] = temp_val[0] + 1
					ret_val[1] = temp_val[1]
					ret_val[1][coins.index(coin)] += 1
		return ret_val[0], ret_val[1]


def bottom_up_coins(val, coins):
	"""Bottom up implementation of coin change"""

	# Initializing 2d value array
	# coins_needed = [0 for i in range(len(coins))]
	values = [[0 for i in range(len(coins))]]

	def bottom_up_coins_helper(value):
		"""bottom_up_coins_helper function to fill array table with values"""

		# Start from the bottom and cycle up
		for i in range(1, val + 1):
			coins_needed = [0 for i in range(len(coins))]

			# Base Case if i is a coin just return 1
			if i in coins:
				coins_needed[coins.index(i)] += 1
				values.append(coins_needed)

			# Otherwise look back and see what values can be pulled out
			else:
				min_count = []
				curr_coin = 0
				for coin in coins:
					if coin < i:
						if sum(min_count) > sum(values[i - coin]) or sum(min_count) == 0:
							min_count = values[i - coin]
							curr_coin = coin

				next_count = min_count.copy()
				next_count[coins.index(curr_coin)] += 1
				values.append(next_count)

	# Call the helper function and return the updated values
	bottom_up_coins_helper(val)
	return sum(values[-1]), values[-1]


def greedy_coins(val, coins):
	"""Implement Greedy algorithm for coin change"""

	# Reversing array for greedy algorithm
	coins.sort()
	coins.reverse()
	values = [0 for i in range(len(coins))]
	remaining = val
	i = 0
	total_coins = 0

	# Cycle for each coin going biggest to smallest
	for coin in coins:
		if coin <= remaining:
			total_coins += remaining // coin
			values[i] = remaining // coin
			remaining %= coin
		i += 1

	values.reverse()
	return total_coins, values


def prob_2(n):
	"""Make plots for PDF"""
	naive_times = [0]
	bottom_times = [0]
	domain = [x for x in range(n + 1)]
	coins = [1, 5, 10, 25, 50, 100]
	stop_naive = False

	for k in range(n):
		# Time the naive_coins
		if not stop_naive:
			start = time.time()
			print(naive_coins(k, coins))
			finish = time.time()
			naive = finish - start
			print("Naive Time:", naive)
			if naive > 60:
				stop_naive = True

		# Time the bottom_up_coins
		start = time.time()
		print(bottom_up_coins(k, coins))
		finish = time.time()
		bottom = finish - start
		print("Bottom Time:", bottom)

		# Append times
		naive_times.append(naive)
		bottom_times.append(bottom)

	# Plot each graph on one plot
	plt.plot(domain, naive_times, 'g-', label="Naive")
	plt.plot(domain, bottom_times, 'r-', label="Bottom-Up")
	plt.xlabel("n values")
	plt.ylabel("Times")
	plt.title("Problem 2: Coin Counter")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.show()


def prob_3(n):
	"""Make plots for PDF"""
	naive_times = [0]
	bottom_times = [0]
	greedy_times = [0]
	domain = [x for x in range(n + 1)]
	coins = [1, 5, 10, 25, 50, 100]
	stop_naive = False

	for k in range(n):
		# Time the naive_coins
		if not stop_naive:
			start = time.time()
			print(naive_coins(k, coins))
			finish = time.time()
			naive = finish - start
			print("Naive Time:", naive)
			if naive > 60:
				stop_naive = True

		# Time the bottom_up_coins
		start = time.time()
		print(bottom_up_coins(k, coins))
		finish = time.time()
		bottom = finish - start
		print("Bottom Time:", bottom)

		# Time the greedy_coins
		start = time.time()
		print(greedy_coins(k, coins))
		finish = time.time()
		greedy = finish - start
		print("Greedy Time:", bottom)

		# Append times
		naive_times.append(naive)
		bottom_times.append(bottom)
		greedy_times.append(greedy)

	# Plot each graph on one plot
	plt.plot(domain, naive_times, 'g-', label="Naive")
	plt.plot(domain, bottom_times, 'r-', label="Bottom-Up")
	plt.plot(domain, greedy_times, 'b-', label="Greedy")
	plt.xlabel("n values")
	plt.ylabel("Times")
	plt.title("Problem 3: Coin Counter w/Greedy")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	prob_1(42)
	# prob_2(2000)
	# prob_3(2000)
