import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	a = np.array(a)
	prod1 = a.shape[0] * a.shape[1]
	prod2 = 1
	for t in new_shape:
		prod2 *= t
	if prod1 != prod2:
		return []
	reshaped_matrix = np.reshape(a, new_shape).tolist()
	return reshaped_matrix