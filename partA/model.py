import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_conv_layers, num_filters, filter_size, activation, dense_layer_neurons):
        super(CNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dense_layer_neurons = dense_layer_neurons
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # Assuming input images are RGB
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, num_filters, filter_size, padding=filter_size//2))
            self.conv_layers.append(activation())
            self.conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = num_filters
        
        # Assuming the output size of the last conv layer is 
        # num_filters * (input_size / (2 ** num_conv_layers)) ** 2
        #  For simplicity, let's assume input_size = 224 (adjust as needed)
        conv_output_size = num_filters * (224 // (2 ** num_conv_layers)) ** 2
        
        self.dense = nn.Linear(conv_output_size, dense_layer_neurons)
        self.output = nn.Linear(dense_layer_neurons, 10)  # 10 output classes

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        x = self.output(x)
        return x

    def calculate_complexity(self, input_size=224):
        # Calculate total computations and parameters
        total_computations = 0
        total_parameters = 0
        
        in_channels = 3
        out_size = input_size
        for i in range(self.num_conv_layers):
            # Conv layer computations: (in_channels * filter_size^2 + 1) * out_channels * out_size^2
            total_computations += (in_channels * self.filter_size**2 + 1) * self.num_filters * out_size**2
            # Conv layer parameters: (in_channels * filter_size^2 + 1) * out_channels
            total_parameters += (in_channels * self.filter_size**2 + 1) * self.num_filters
            in_channels = self.num_filters
            out_size //= 2  # Max pooling reduces size by half
            
        # Dense layer computations: in_features * out_features
        total_computations += self.num_filters * (224 // (2 ** self.num_conv_layers)) ** 2 * self.dense_layer_neurons
        # Dense layer parameters: in_features * out_features + out_features (bias)
        total_parameters += self.num_filters * (224 // (2 ** self.num_conv_layers)) ** 2 * self.dense_layer_neurons + self.dense_layer_neurons
        
        # Output layer computations: in_features * out_features
        total_computations += self.dense_layer_neurons * 10
        # Output layer parameters: in_features * out_features + out_features (bias)
        total_parameters += self.dense_layer_neurons * 10 + 10

        return total_computations, total_parameters
