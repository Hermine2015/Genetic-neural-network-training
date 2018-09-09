
class ConvolutionalConfiguration:
    def __init__(self,
                 total_convolutional_layers,
                 total_convolutional_filters,
                 filter_size_convolution,
                 filter_size_deconvolution,
                 pool_size,
                 strides,
                 activation,
                 padding,
                 output_activation,
                 optimizer,
                 loss
                 ):
        self.total_convolutional_layers = total_convolutional_layers
        self.total_convolutional_filters = total_convolutional_filters
        self.filter_size_convolution = filter_size_convolution
        self.filter_size_deconvolution = filter_size_deconvolution
        self.pool_size = pool_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss

