import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_diagnostics():
    df_path = "output/synthetic_datasets_noise_1_10000_details.csv"
    if not os.path.exists(df_path):
        print(f"File {df_path} not found.")
        return
    
    df = pd.read_csv(df_path)
    
    # Filter for CVAP models to see the containment reliability
    cvap_df = df[df['model'].str.contains('CVAP')].copy()
    
    # We want to show how RMSE (point predictor accuracy) correlates with Containment (interval reliability)
    # AND how Weight MSE (linear recovery) correlates with Containment.
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: RMSE vs Mean Containment
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=cvap_df, x='rmse', y='mean_containment', hue='scenario', alpha=0.6)
    plt.title('RMSE vs Mean Containment (Synthetic CVAP)')
    plt.xlabel('RMSE (Point Predictor Error)')
    plt.ylabel('Mean Containment of $E[Y|X]$')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Weight MSE vs Mean Containment (Only if weight_mse exists)
    plt.subplot(1, 2, 2)
    if 'weight_mse' in cvap_df.columns:
        linear_df = cvap_df.dropna(subset=['weight_mse'])
        if not linear_df.empty:
            sns.scatterplot(data=linear_df, x='weight_mse', y='mean_containment', hue='scenario', alpha=0.6)
            plt.title('Weight MSE vs Mean Containment (Linear Scenarios)')
            plt.xlabel('Weight MSE (Coefficient Recovery Error)')
            plt.ylabel('Mean Containment of $E[Y|X]$')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'No Weight MSE data available', ha='center')
    else:
        plt.text(0.5, 0.5, 'Weight MSE column missing', ha='center')
        
    plt.tight_layout()
    plt.savefig('output/diagnostic_plots.png', dpi=300)
    print("Plots saved to output/diagnostic_plots.png")

if __name__ == "__main__":
    plot_diagnostics()
