class MinMaxScaler:
    def __call__(self, array):
        scale = 1.0 / (array.max() - array.min())
        array = array * scale - array.min() * scale
        return array
