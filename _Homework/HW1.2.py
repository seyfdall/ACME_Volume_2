def argmin(num_list):
	min_el = num_list[0]
	min_index = 0
	i = 1
	while i < len(num_list):
		curr_el = num_list[i]
		if curr_el < min_el:
			min_index = i
			min_el = curr_el
		i += 1
	return min_index


if __name__ == "__main__":
	print(argmin([3, 2, 1, 4, 5]))
