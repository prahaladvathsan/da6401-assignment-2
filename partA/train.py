import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import wandb
from multiprocessing import freeze_support
import os
import numpy as np
from model import CNN

def get_sweep_config():
    """
    Define the sweep configuration for hyperparameter tuning
    """
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization for efficient search
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'num_conv_layers': {
                'values': [3, 4, 5]  # As per assignment, we need 5 conv layers, but can try fewer
            },
            'num_filters': {
                'values': [16, 32, 64]  # Different filter configurations
            },
            'filter_size': {
                'values': [3, 5]  # Different filter sizes
            },
            'activation': {
                'values': ['ReLU', 'GELU', 'SiLU']  # Different activation functions
            },
            'dense_layer_neurons': {
                'values': [64, 128, 256]  # Different numbers of neurons in dense layer
            },
            'learning_rate': {
                'values': [1e-4, 5e-4, 1e-3, 5e-3]
            },
            'batch_norm': {
                'values': [True, False]  # Whether to use batch normalization
            },
            'dropout_rate': {
                'values': [0.0, 0.3, 0.5]  # Different dropout rates
            },
            'data_augmentation': {
                'values': [True, False]  # Whether to use data augmentation
            }
        }
    }
    return sweep_config

# Define the LightningModule for training
class LitCNN(pl.LightningModule):
    def __init__(self, num_conv_layers, num_filters, filter_size, activation, dense_layer_neurons, 
                learning_rate=1e-3, batch_norm=False, dropout_rate=0.0, data_augmentation=False, input_size=128):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the model
        self.model = CNN(
            num_conv_layers=num_conv_layers, 
            num_filters=num_filters, 
            filter_size=filter_size, 
            activation=activation, 
            dense_layer_neurons=dense_layer_neurons,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            input_size=input_size
        )
        
        self.learning_rate = learning_rate
        self.data_augmentation = data_augmentation
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy for monitoring
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Define the data transforms with optional augmentation
def get_transforms(data_augmentation=False, input_size=128):
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Function for running a sweep
def train_sweep_model():
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    # Set up training with the config parameters
    input_size = 128  # Fixed input size for consistent dimensions
    train_model(
        num_conv_layers=config.num_conv_layers,
        num_filters=config.num_filters,
        filter_size=config.filter_size,
        activation=config.activation,
        dense_layer_neurons=config.dense_layer_neurons,
        learning_rate=config.learning_rate,
        batch_norm=config.batch_norm,
        dropout_rate=config.dropout_rate,
        data_augmentation=config.data_augmentation,
        input_size=input_size,
        subset_fraction=0.3,  # Use 30% of data for faster sweep
        max_epochs=10,
        batch_size=16
    )

# Main training function
def train_model(
    num_conv_layers=3,
    num_filters=32,
    filter_size=3,
    activation='ReLU',
    dense_layer_neurons=64,
    learning_rate=1e-3,
    batch_norm=True,
    dropout_rate=0.3,
    data_augmentation=True,
    input_size=128,
    subset_fraction=1.0,
    max_epochs=10,
    batch_size=32
):
    """
    Train a CNN model with the specified parameters.
    """
    # If not in a wandb run, initialize one
    if wandb.run is None:
        wandb.init(project='da6401_assignment2_partA')
    
    # Log hyperparameters
    wandb.config.update({
        'num_conv_layers': num_conv_layers,
        'num_filters': num_filters,
        'filter_size': filter_size,
        'activation': activation,
        'dense_layer_neurons': dense_layer_neurons,
        'learning_rate': learning_rate,
        'batch_norm': batch_norm,
        'dropout_rate': dropout_rate,
        'data_augmentation': data_augmentation,
        'input_size': input_size,
        'subset_fraction': subset_fraction,
        'max_epochs': max_epochs,
        'batch_size': batch_size
    })
    
    # Get data transforms
    train_transform, val_transform = get_transforms(
        data_augmentation=data_augmentation,
        input_size=input_size
    )
    
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(root='data/train_split', transform=train_transform)
        val_dataset = datasets.ImageFolder(root='data/val_split', transform=val_transform)
        
        # Create subset for faster training if needed
        if subset_fraction < 1.0:
            train_size = len(train_dataset)
            indices = np.random.choice(train_size, size=int(train_size * subset_fraction), replace=False)
            train_dataset = Subset(train_dataset, indices)
            print(f"Using {len(indices)} images for training (subset of full dataset)")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Increase worker processes
            pin_memory=True,  # Enable pin memory for faster data transfer
            persistent_workers=True  # Keep workers alive between iterations
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Initialize model
    model = LitCNN(
        num_conv_layers=num_conv_layers,
        num_filters=num_filters,
        filter_size=filter_size,
        activation=activation,
        dense_layer_neurons=dense_layer_neurons,
        learning_rate=learning_rate,
        batch_norm=batch_norm,
        dropout_rate=dropout_rate,
        data_augmentation=data_augmentation,
        input_size=input_size
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=pl.loggers.WandbLogger(),
        accelerator='auto',  # Automatically use GPU if available
        devices='auto',  # Use all available devices
        num_sanity_val_steps=2,  # Run validation twice to check setup
        gradient_clip_val=1.0,  # Add gradient clipping
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_acc:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=5,  # Increase patience slightly
                mode='max'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')  # Monitor learning rate
        ]
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete!")
    
    # Calculate and log model complexity
    with torch.no_grad():
        computations, parameters = model.model.calculate_complexity()
        print(f"Model computations: {computations:,}")
        print(f"Model parameters: {parameters:,}")
        
        wandb.log({
            "total_computations": computations,
            "total_parameters": parameters
        })

def main():
    # Check if we should run a sweep or a single training
    if os.environ.get('RUN_SWEEP', 'False').lower() == 'true':
        print("Running hyperparameter sweep...")
        sweep_id = wandb.sweep(get_sweep_config(), project='da6401_assignment2_partA')
        wandb.agent(sweep_id, train_sweep_model, count=20)  # Run 20 different configurations
    else:
        print("Running single training with default parameters...")
        # Run with default parameters, but use a small subset for faster training on CPU
        train_model(
            num_conv_layers=5,  # Smaller model for CPU
            num_filters=32,
            filter_size=3,
            activation='ReLU',
            dense_layer_neurons=64,
            batch_norm=True,
            dropout_rate=0.3,
            subset_fraction=1,  # Use 30% of data
            max_epochs=10
        )

if __name__ == "__main__":
    freeze_support()  # For Windows support with multiprocessing
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()