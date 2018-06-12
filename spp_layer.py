from keras.engine.topology import Layer

class SpatialPyramidPoling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
        # Input shape
        or 4D tensor with shape:
        (samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        (samples, channels * sum([i * i for i in pool_list])
    """

    def  __init__(self, pool_list, **kwargs):
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i**2 for i in pool_list])
        super(SpatialPyramidPolinhttps, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_channels*self.num_outputs_per_channel)

    def get_config(self):
        config = {
            'pool_list':self.pool_list
        }

        base_config = super(SpatialPyramidPoling, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def call(self, x, mask=None):
        input_shape = K.shape(x)
        num_rows = input_shape[1]
        num_cols = input_shape[2]
        row_lenght = [K.cast(num_rows, 'float32')/i for i in self.pool_list]
        col_lenght = [K.cast(num_cols, 'float32')/i for i in self.pool_list]

        results = []

        for ip, p in enumerate(self.pool_list):
            for j in range(p):
                for i in range(p):
                    x1 = i * col_length[ip]
                    x2 = i * col_length[ip] + col_length[ip]
                    y1 = j * row_length[ip]
                    y2 = j * row_length[ip] + row_length[ip]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    results.append(pooled_val)
        results = K.concatenate(results)
        return results
