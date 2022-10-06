"""
Temporal Complexity => 8n - 1 ~ 8n leading order
Spatial Complexity => n + 4 ~ n leading order
"""
def argmin(num_list): # Space - n slots
	# Getting length of list
	length = len(num_list)  # Time - 2 primitive operations : Space - 1 slot
	# Initializing the minimum element
	min_el = num_list[0]  # Time - 2 primitive operations : Space - 1 slot
	# Initialize the min_index
	min_index = 0  # Time - 2 primitive operations : Space - 1 slot
	# Initialize the indexing variable
	i = 1  # Time - 1 primitive operation : Space - 1 slot
	# Cycle through the list
	while i < length:  # Time - 1 primitive operation (n - 1 times)
		# Check to see if the next num is smaller
		if num_list[i] < min_el:  # Time - 2 primitive operations
			# Set the index
			min_index = i  # Time - 1 primitive operation
			# Update the minimum element
			min_el = num_list[i]  # Time - 2 primitive operations
		# Increment the index
		i += 1  # Time - 2 primitive operations
	# Return the minimum index
	return min_index

"""
Temporal Complexity - 3 + 8n^2 + 10n ~ 8n^2 leading order
Spatial Complexity - 2 + n ~ n leading order
"""
def selection_sort(num_list): # Space - n slots
	# Set variable to track length of list
	length = len(num_list)	# Time - 2 primitive operations : Space 1 slot
	# Set variable to index through the list
	i = 0	# Time - 1 primitive operation : Space 1 slot
	# Cycle through the list
	while i < length:	# Time - 1 primitive operation (n times)
		# Find the next minimum index
		next_min_index = argmin(num_list[i:]) + i	# Time - 8n - 1 primitive operations (list gets shorter) : Space 1 slot
		# Setup a swap variable
		temp = num_list[i]	# Time - 2 primitive operations : Space 1 slot
		# Swap the next minimum with the next spot in the list
		num_list[i] = num_list[next_min_index]	# Time - 3 primitive operations : Space 1 slot
		# Replace the original swapped value slot with the temp
		num_list[next_min_index] = temp	# Time - 3 primitive operations : Space 1 slot
		# Increment the index
		i += 1 # Time - 2 primitive operations : Space 1 slot
	return num_list


if __name__ == "__main__":
	print(argmin([3, 2, 1, 4, 5]))
	print(selection_sort([4, 3, 2, 1, 10]))
