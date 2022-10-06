def max_unimodal(seq, left = 0, right = None):
	"""A Binary Search style algorithm which takes in a list and two pointers and recursively
	cycles until it returns the max of the list - runs in O(logn) time
	"""

	# Starting out set right to be the last index if it's not set
	if right is None:
		right = len(seq) - 1
	mid = (left + right) // 2

	# Base cases to return the max of the list or -1 if it doesn't exist
	if left > right:
		return -1
	elif right - left == 1:
		if seq[right] > seq[left]:
			return seq[right]
		else:
			return seq[left]
	elif right == left:
		return seq[left]

	# Recursive cases to rotate to the left or right of the list depending on which one is greater than the mid
	if seq[mid - 1] > seq[mid]:
		return max_unimodal(seq, left, mid - 1)
	else:
		return max_unimodal(seq, mid, right)


if __name__ == "__main__":
	print(max_unimodal([0, 1, 2, 3, 2, 1, 0]))
	print(max_unimodal([0, 1, 2, 3, 4, 5]))
	print(max_unimodal([5, 4, 3, 2, 1, 0]))
	print(max_unimodal([1, 8]))
	print(max_unimodal([0]))
	print(max_unimodal([1,2,3,3,3,2,1]))
