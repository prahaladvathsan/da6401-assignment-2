# Part A: Training a CNN from Scratch

This directory contains code for training a CNN model from scratch on a subset of the iNaturalist dataset.

## Requirements

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Directory Structure

The dataset should be organized as follows:
```
data/
├── train_split/        # 80% of training data
│   ├── class1/         # Images for class 1
│   ├── class2/         # Images for class 2
│   └── ...             # Other classes
├── val_split/          # 20% of training data
│   ├── class1/         # Images for class 1
│   ├── class2/         # Images for class 2
│   └── ...             # Other classes
└── test/               # Test data
    ├── class1/         # Images for class 1
    ├── class2/         # Images for class 2
    └── ...             # Other classes
```

## Files

- `model.py`: Contains the CNN model architecture
- `train.py`: Code for training the model
- `evaluate.py`: Code for evaluating the trained model
- `sweep_analysis.py`: Code for analyzing hyperparameter sweep results

## Training the Model

### Single Training Run

To train the model with default hyperparameters:

```bash
python train.py
```

### Hyperparameter Sweep

To run a hyperparameter sweep with wandb:

1. Make sure you have a wandb account and have logged in:
   ```bash
   wandb login
   ```

2. Uncomment the sweep code in `train.py` and run:
   ```bash
   python train.py
   ```

3. After the sweep is complete, analyze the results:
   ```bash
   python sweep_analysis.py
   ```
   
   Note: Make sure to update the sweep_id in `sweep_analysis.py` with your actual sweep ID.

## Evaluating the Model

To evaluate the trained model on the test set:

1. Update the checkpoint path in `evaluate.py` to point to your best model checkpoint.
2. Run:
   ```bash
   python evaluate.py
   ```

## Model Architecture

The CNN model consists of:
- Configurable number of convolutional layers (default: 5)
- Each convolutional layer is followed by activation and max-pooling
- Optional batch normalization after each convolution
- A dense layer after the convolutional layers
- Optional dropout for regularization
- An output layer with 10 neurons (for 10 classes)

Hyperparameters that can be configured:
- Number of convolutional layers
- Number of filters in each layer
- Filter size
- Activation function
- Number of neurons in the dense layer
- Learning rate
- Whether to use batch normalization
- Dropout rate
- Whether to use data augmentation

## Complexity Analysis

The `calculate_complexity` method in the CNN class calculates:
- Total number of computations
- Total number of parameters

For a model with:
- m filters in each layer
- filter size of k×k
- n neurons in the dense layer

## Results

After running the hyperparameter sweep, the best model achieved [XX]% accuracy on the validation set and [YY]% accuracy on the test set.

Key insights from the hyperparameter sweep:
1. [Your insight about filter count]
2. [Your insight about filter size]
3. [Your insight about activation function]
4. [Your insight about batch normalization]
5. [Your insight about dropout]
6. [Your insight about data augmentation]