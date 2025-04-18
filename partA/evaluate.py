import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from model import CNN  # Assuming model.py is in the same directory

# Load the trained model (replace with the actual path to your best model checkpoint)
model = LitCNN.load_from_checkpoint('path/to/your/best/model.ckpt')
model.eval()

# Data loading and transforms for test data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add any necessary transforms (e.g., normalization)
])

# Load the iNaturalist test dataset (replace with your actual path)
test_dataset = datasets.ImageFolder(root='data/inaturalist_test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

# Evaluate on the test set
trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
test_results = trainer.test(model, test_loader)
print(f'Test Results: {test_results}')

# Generate and display a 10x3 grid of sample images and predictions
num_images = 30

# Get a batch of test data
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Get predictions
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Plot the images and predictions
fig, axes = plt.subplots(10, 3, figsize=(12, 30))
axes = axes.flatten()

for i in range(num_images):
    # Convert image from tensor to numpy and transpose dimensions for display
    img = images[i].numpy().transpose((1, 2, 0))
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f'True: {labels[i].item()}, Predicted: {predicted[i].item()}')  # Assuming class indices as labels
    ax.axis('off')

plt.tight_layout()
plt.show()
