import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np
import argparse

def analyze_sweep(entity, project, sweep_id, output_dir='./'):
    """
    Analyze the results of a wandb sweep and generate insights.
    
    Args:
        entity (str): The wandb entity (username)
        project (str): The wandb project name
        sweep_id (str): The sweep ID to analyze
        output_dir (str): Directory to save the plots and results
    """
    # Initialize wandb and get the sweep results
    api = wandb.Api()
    
    # Get all runs from the sweep
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    
    print(f"Found {len(runs)} runs in sweep {sweep_id}")
    
    # Create a dataframe with all the runs and their configurations/results
    runs_df = pd.DataFrame()
    for run in runs:
        # Get run details
        run_data = {
            'run_id': run.id,
            'run_name': run.name,
            'val_acc': run.summary.get('val_acc', 0),  # Use 0 if val_acc not found
            'val_loss': run.summary.get('val_loss', float('inf')),  # Use inf if val_loss not found
            'num_conv_layers': run.config.get('num_conv_layers', None),
            'num_filters': run.config.get('num_filters', None),
            'filter_size': run.config.get('filter_size', None),
            'activation': run.config.get('activation', None),
            'dense_layer_neurons': run.config.get('dense_layer_neurons', None),
            'learning_rate': run.config.get('learning_rate', None),
            'batch_norm': run.config.get('batch_norm', None),
            'dropout_rate': run.config.get('dropout_rate', None),
            'data_augmentation': run.config.get('data_augmentation', None),
        }
        runs_df = pd.concat([runs_df, pd.DataFrame([run_data])], ignore_index=True)
    
    # Sort by validation accuracy (highest first)
    runs_df = runs_df.sort_values('val_acc', ascending=False)
    
    # Print the top 5 best performing configurations
    print("\n--- Top 5 Best Performing Configurations ---")
    top_5 = runs_df.head(5)
    for i, row in top_5.iterrows():
        print(f"Run {i+1}: Accuracy = {row['val_acc']:.4f}")
        print(f"  Layers: {row['num_conv_layers']}, Filters: {row['num_filters']}, Filter Size: {row['filter_size']}")
        print(f"  Activation: {row['activation']}, Dense Neurons: {row['dense_layer_neurons']}")
        print(f"  Learning Rate: {row['learning_rate']:.6f}, Batch Norm: {row['batch_norm']}")
        print(f"  Dropout: {row['dropout_rate']}, Data Augmentation: {row['data_augmentation']}\n")
    
    # Create accuracy vs. runs plot (created plot)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(runs_df)), runs_df['val_acc'], 'o-', alpha=0.7)
    plt.title('Validation Accuracy vs. Experiment Run')
    plt.xlabel('Experiment Run Index')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/accuracy_vs_runs.png')
    print(f"Saved accuracy plot to {output_dir}/accuracy_vs_runs.png")
    plt.close()
    
    # Create a correlation matrix for hyperparameters vs. validation accuracy
    # First, convert categorical variables to numeric
    categorical_cols = ['activation', 'batch_norm', 'data_augmentation']
    for col in categorical_cols:
        if col in runs_df.columns:
            runs_df[col] = runs_df[col].astype('category').cat.codes
    
    # Calculate correlation
    corr_cols = ['val_acc', 'val_loss', 'num_conv_layers', 'num_filters', 'filter_size', 
                 'dense_layer_neurons', 'learning_rate', 'dropout_rate']
    if 'batch_norm' in runs_df.columns:
        corr_cols.append('batch_norm')
    if 'data_augmentation' in runs_df.columns:
        corr_cols.append('data_augmentation')
    if 'activation' in runs_df.columns:
        corr_cols.append('activation')
    
    # Remove columns that are all None
    valid_cols = [col for col in corr_cols if not runs_df[col].isnull().all()]
    corr_df = runs_df[valid_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation between Hyperparameters and Performance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    print(f"Saved correlation heatmap to {output_dir}/correlation_heatmap.png")
    plt.close()
    
    # Create parallel coordinates plot
    # First, normalize the data for better visualization
    columns_to_normalize = ['num_filters', 'filter_size', 'dense_layer_neurons', 'learning_rate', 'dropout_rate']
    normalized_df = runs_df.copy()
    for col in columns_to_normalize:
        if col in normalized_df.columns and not normalized_df[col].isnull().all():
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    # Add color based on validation accuracy (higher is better)
    norm = plt.Normalize(normalized_df['val_acc'].min(), normalized_df['val_acc'].max())
    colors = plt.cm.viridis(norm(normalized_df['val_acc']))
    
    # Select columns for parallel coordinates plot
    parallel_cols = ['num_conv_layers', 'num_filters', 'filter_size', 'dense_layer_neurons', 
                    'learning_rate', 'dropout_rate', 'val_acc']
    if 'batch_norm' in normalized_df.columns:
        parallel_cols.append('batch_norm')
    if 'data_augmentation' in normalized_df.columns:
        parallel_cols.append('data_augmentation')
    
    # Remove columns that are all None
    valid_parallel_cols = [col for col in parallel_cols if not normalized_df[col].isnull().all()]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    pd.plotting.parallel_coordinates(
        normalized_df[valid_parallel_cols], 
        'val_acc', 
        color=colors,
        alpha=0.5
    )
    plt.title('Parallel Coordinates Plot of Hyperparameters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parallel_coordinates.png')
    print(f"Saved parallel coordinates plot to {output_dir}/parallel_coordinates.png")
    plt.close()
    
    # Generate insights about the hyperparameters
    
    # Best hyperparameter configuration
    best_config = runs_df.iloc[0]
    print("\n--- Best Hyperparameter Configuration ---")
    for col in ['num_conv_layers', 'num_filters', 'filter_size', 'activation', 
                'dense_layer_neurons', 'learning_rate', 'batch_norm', 
                'dropout_rate', 'data_augmentation']:
        if col in best_config and not pd.isna(best_config[col]):
            print(f"{col}: {best_config[col]}")
    
    # Analyze the impact of each hyperparameter
    print("\n--- Analysis of Hyperparameter Impact ---")
    
    # Impact of number of filters
    if 'num_filters' in runs_df.columns and not runs_df['num_filters'].isnull().all():
        filter_analysis = runs_df.groupby('num_filters')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by number of filters:")
        for _, row in filter_analysis.iterrows():
            print(f"Filters: {row['num_filters']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Impact of filter size
    if 'filter_size' in runs_df.columns and not runs_df['filter_size'].isnull().all():
        filter_size_analysis = runs_df.groupby('filter_size')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by filter size:")
        for _, row in filter_size_analysis.iterrows():
            print(f"Filter Size: {row['filter_size']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Impact of activation function
    if 'activation' in runs_df.columns and not runs_df['activation'].isnull().all():
        activation_analysis = runs_df.groupby('activation')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by activation function:")
        for _, row in activation_analysis.iterrows():
            print(f"Activation: {row['activation']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Impact of batch normalization
    if 'batch_norm' in runs_df.columns and not runs_df['batch_norm'].isnull().all():
        batch_norm_analysis = runs_df.groupby('batch_norm')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by batch normalization:")
        for _, row in batch_norm_analysis.iterrows():
            print(f"Batch Norm: {row['batch_norm']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Impact of dropout rate
    if 'dropout_rate' in runs_df.columns and not runs_df['dropout_rate'].isnull().all():
        dropout_analysis = runs_df.groupby('dropout_rate')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by dropout rate:")
        for _, row in dropout_analysis.iterrows():
            print(f"Dropout Rate: {row['dropout_rate']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Impact of data augmentation
    if 'data_augmentation' in runs_df.columns and not runs_df['data_augmentation'].isnull().all():
        augmentation_analysis = runs_df.groupby('data_augmentation')['val_acc'].mean().reset_index()
        print(f"\nAverage validation accuracy by data augmentation:")
        for _, row in augmentation_analysis.iterrows():
            print(f"Data Augmentation: {row['data_augmentation']}, Avg Accuracy: {row['val_acc']:.4f}")
    
    # Write results to a markdown file for easy inclusion in reports
    with open(f'{output_dir}/sweep_analysis_results.md', 'w') as f:
        f.write("# Hyperparameter Sweep Analysis Results\n\n")
        
        f.write("## Best Hyperparameter Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| --- | --- |\n")
        for param in ['num_conv_layers', 'num_filters', 'filter_size', 'activation', 
                     'dense_layer_neurons', 'learning_rate', 'batch_norm', 
                     'dropout_rate', 'data_augmentation']:
            if param in best_config and not pd.isna(best_config[param]):
                f.write(f"| {param} | {best_config[param]} |\n")
        
        f.write(f"\n## Validation Accuracy: {best_config['val_acc']:.4f}\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("After analyzing the hyperparameter sweep results, the following insights were observed:\n\n")
        f.write("1. [Your insight about filter count here]\n")
        f.write("2. [Your insight about filter size here]\n")
        f.write("3. [Your insight about activation function here]\n")
        f.write("4. [Your insight about batch normalization here]\n")
        f.write("5. [Your insight about dropout here]\n")
        f.write("6. [Your insight about data augmentation here]\n")
        f.write("7. [Other insight based on correlation analysis]\n")
        f.write("8. [Other insight based on parallel coordinates plot]\n\n")
        
        f.write("## Plots\n\n")
        f.write("### Accuracy vs. Experiment Run\n\n")
        f.write("![Accuracy vs. Runs](accuracy_vs_runs.png)\n\n")
        
        f.write("### Correlation Heatmap\n\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        
        f.write("### Parallel Coordinates Plot\n\n")
        f.write("![Parallel Coordinates](parallel_coordinates.png)\n\n")
    
    print(f"\nAnalysis complete. Check {output_dir}/sweep_analysis_results.md for a report template.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze wandb sweep results')
    parser.add_argument('--entity', type=str, required=True, help='wandb entity (username)')
    parser.add_argument('--project', type=str, required=True, help='wandb project name')
    parser.add_argument('--sweep_id', type=str, required=True, help='wandb sweep ID')
    parser.add_argument('--output_dir', type=str, default='./', help='directory to save plots and results')
    
    args = parser.parse_args()
    analyze_sweep(args.entity, args.project, args.sweep_id, args.output_dir)