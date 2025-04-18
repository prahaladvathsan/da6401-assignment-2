import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import wandb
from model import create_pretrained_model  # Assuming model.py is in the same directory

# Define a LightningModule for fine-tuning
class LitPretrainedCNN(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-3, pretrained=True, freeze_layers=False, freeze_strategy='last_layer'):
        super().__init__()
        self.model = create_pretrained_model(model_name, num_classes, pretrained, freeze_layers, freeze_strategy)
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
    transforms.Resize((224, 224)),
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
wandb.init(project='da6401_assignment2_partB')

# --- Example of different fine-tuning strategies ---
# Strategy 1: Fine-tune all layers
# model = LitPretrainedCNN(
#     model_name='resnet50',
#     num_classes=10,
#     learning_rate=1e-4,
#     pretrained=True,
#     freeze_layers=False
# )
# wandb.config.update({'strategy': 'all_layers'})

# Strategy 2: Fine-tune only the last layer
model = LitPretrainedCNN(
    model_name='resnet50',
    num_classes=10,
    learning_rate=1e-3,
    pretrained=True,
    freeze_layers=True,
    freeze_strategy='last_layer'
)
wandb.config.update({'strategy': 'last_layer'})

# Strategy 3: Fine-tune with a different learning rate
# model = LitPretrainedCNN(
#     model_name='resnet50',
#     num_classes=10,
#     learning_rate=1e-5,  # Lower learning rate
#     pretrained=True,
#     freeze_layers=True,
#     freeze_strategy='last_layer'
# )
# wandb.config.update({'strategy': 'last_layer_lr_1e-5'})

# Log hyperparameters (including the chosen strategy)
wandb.config.update({
    'model_name': 'resnet50',
    'num_classes': 10,
    'learning_rate': model.learning_rate,
    'pretrained': True,
    'freeze_layers': model.model.freeze_layers if hasattr(model.model, 'freeze_layers') else False,
    'freeze_strategy': model.model.freeze_strategy if hasattr(model.model, 'freeze_strategy') else 'None'
})

# Training
trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.WandbLogger(), accelerator='gpu' if torch.cuda.is_available() else 'cpu')  # Adjust max_epochs
trainer.fit(model, train_loader, val_loader)

# Finish wandb run
wandb.finish()
