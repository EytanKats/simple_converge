import numpy as np


class Array(object):

    @staticmethod
    def format_array(array, formatting):

        el_formatting = "{0:" + formatting + "}"
        v_func = np.vectorize(lambda el: "".join(el_formatting.format(el)))  # create vectorized function
        formatted_array = v_func(array)  # apply function on each element in array

        return formatted_array
