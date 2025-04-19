import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_conv_layers, num_filters, filter_size, activation, dense_layer_neurons, 
                 batch_norm=False, dropout_rate=0.0, input_size=128):
        super().__init__()
        self.features = nn.ModuleList()
        
        # Input channels for first conv layer
        in_channels = 3
        out_size = input_size
        
        # Create feature extraction layers
        for i in range(num_conv_layers):
            # Conv layer with padding to maintain size
            self.features.append(
                nn.Conv2d(in_channels, num_filters, filter_size, padding=filter_size//2)
            )
            
            # Always add batch norm after conv (helps with training)
            self.features.append(nn.BatchNorm2d(num_filters))
            
            # Add activation
            if isinstance(activation, str):
                if activation == 'ReLU':
                    self.features.append(nn.ReLU(inplace=True))
                elif activation == 'GELU':
                    self.features.append(nn.GELU())
                elif activation == 'SiLU':
                    self.features.append(nn.SiLU(inplace=True))
                elif activation == 'Mish':
                    self.features.append(nn.Mish(inplace=True))
            else:
                self.features.append(activation())
            
            # Add max pooling every 2 layers
            if (i + 1) % 2 == 0:
                self.features.append(nn.MaxPool2d(2))
                out_size = out_size // 2
            
            in_channels = num_filters
        
        self.flattened_size = num_filters * out_size * out_size
        
        # Create classifier layers with better initialization
        layers = []
        layers.append(nn.Linear(self.flattened_size, dense_layer_neurons))
        nn.init.kaiming_normal_(layers[-1].weight)  # He initialization
        
        # Add batch norm after dense layer
        if batch_norm:
            layers.append(nn.BatchNorm1d(dense_layer_neurons))
        
        # Add activation
        if isinstance(activation, str):
            if activation == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'GELU':
                layers.append(nn.GELU())
            elif activation == 'SiLU':
                layers.append(nn.SiLU(inplace=True))
            elif activation == 'Mish':
                layers.append(nn.Mish(inplace=True))
        else:
            layers.append(activation())
        
        # Add dropout if specified
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer with better initialization
        layers.append(nn.Linear(dense_layer_neurons, 10))  # 10 output classes
        nn.init.kaiming_normal_(layers[-1].weight)
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Pass through feature extraction layers
        for layer in self.features:
            x = layer(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Pass through classifier
        x = self.classifier(x)
        return x

    def calculate_complexity(self):
        # Helper function to calculate model complexity
        computations = 0
        parameters = 0
        
        # Count conv operations
        x = torch.randn(1, 3, 128, 128)  # Dummy input
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                # Count MACs for conv layer
                computations += (
                    layer.in_channels * 
                    layer.out_channels * 
                    layer.kernel_size[0] * 
                    layer.kernel_size[1] * 
                    x.shape[2] * 
                    x.shape[3]
                )
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)  # Update size after pooling
        
        # Count parameters
        parameters = sum(p.numel() for p in self.parameters())
        
        return computations, parameters