import numpy as np

### Q1


class convolution():              
    def __init__(self, n_filters, kernel_size, padding, stride, activation=None):              
        self.output_channel = n_filters
        self.filter_size = kernel_size[0]

        if padding == "VALID":
                self.pad = 0
        else:
            self.pad = (self.filter_size -1)//2
        
        self.stride = stride
        self.activation = activation
    
    def forward(self, X): #, figures=None):
        self.input = X
        self.batch_size = X.shape[0]
        self.input_size = X.shape[1]
        self.input_channel = X.shape[3]

        self.filters = np.random.randn(self.output_channel, self.filter_size, self.filter_size, self.input_channel)

        output_dim = int(((self.input.shape[1]-self.filter_size + 2 * self.pad)// self.stride)+1)
        output_image = np.zeros((self.batch_size, output_dim, output_dim, self.output_channel))

        X_padded = np.pad (X, ((0,0), (self.pad, self.pad), (self.pad, self.pad), (0,0)), 'constant', constant_values=0)

        for r in range(self.batch_size):
            for k in range(self.output_channel):
                filter = self.filters[k]
                for i in range(output_dim): #horizontal axis
                    for j in range(output_dim): # vertical axis
                        output_image[r,i,j,k] = np.multiply(filter, X_padded[r,i* self.stride + self.filter_size, 
                                                            j * self.stride: j * self.stride + self.filter_size, :]).sum()


### Q2 

# outputOfEachConvLayer = [(in_size + 2*padding - kernel_size) / stride] + 1

def X_flatten(self, X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    X_padded = np.pad (X, ((0,0), (padding, padding), (padding, padding), (0,0)), 'constant', constant_values=0)

    windows = []
    for i in range(out_h):
        for j in range(out_w):
            window = X_padded[:,i *stride:i * stride + window_h, j * stride: j*stride + window_w, :]
            windows.append(window)
    stacked = np.stack(windows)
    return np.reshape(stacked, (-1, window_c * window_w * window_h))

def convolution(X, n_filters, kernel_size, padding, stride):

    global conv_activation_layer
    k_h = kernel_size[0]
    k_w = kernel_size[1]

    if padding == 'VALID':
        pad = 0
    else:
        pad = 1
    
    filters = []
    for i in range(n_filters):
        kernel = np.random.randn(k_h, k_w, X.shape[3])
        filters.append(kernel)
    kernel = np.reshape(filters, (k_h, k_w, X.shape[3], n_filters))

    n,h,w,c = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    filter_h, filter_w, filter_c, filter_n = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    X_flat = model.X_flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, pad)
    W_flat = np.reshape(kernel, (filter_h * filter_w * filter_c, filter_n))

    z = np.matmul(X_flat, W_flat)
    z = np.transpose(np.reshape(z, (out_h, out_w, n, filter_n)), (2,0,1,3))

    conv_activation_layer = relu.activation(z)

    return conv_activation_layer


