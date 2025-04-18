import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import wandb
from model import CNN  # Assuming model.py is in the same directory

# Define a LightningModule for training
class LitCNN(pl.LightningModule):
    def __init__(self, num_conv_layers, num_filters, filter_size, activation, dense_layer_neurons, learning_rate=1e-3):
        super().__init__()
        self.model = CNN(num_conv_layers, num_filters, filter_size, activation, dense_layer_neurons)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Data loading and transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),
    # Add more transforms as needed (e.g., normalization)
])

# Load the iNaturalist dataset (replace with your actual path)
dataset = datasets.ImageFolder(root='data/inaturalist', transform=transform)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

# Initialize wandb
wandb.init(project='da6401_assignment2_partA')

# Initialize the model
model = LitCNN(
    num_conv_layers=3,  # Example hyperparameters, adjust as needed
    num_filters=32,
    filter_size=3,
    activation=nn.ReLU,
    dense_layer_neurons=128,
    learning_rate=1e-3
)

# Log hyperparameters to wandb
wandb.config.update({
    'num_conv_layers': 3,
    'num_filters': 32,
    'filter_size': 3,
    'activation': 'ReLU',
    'dense_layer_neurons': 128,
    'learning_rate': 1e-3
})

# Training
trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.WandbLogger(), accelerator='gpu' if torch.cuda.is_available() else 'cpu')  # Adjust max_epochs
trainer.fit(model, train_loader, val_loader)

# Finish wandb run
wandb.finish()
