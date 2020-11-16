import numpy as np


class MatrixOperator:
    def as_vector(self, elements):
        return np.array([n for n in elements])

    def as_matrix(self, elements, line_length):
        return np.array([elements[i:i + line_length] for i in range(0, len(elements), line_length)])

    def dot_operator(self, first_element, second_element):
        return np.dot(first_element, second_element)
