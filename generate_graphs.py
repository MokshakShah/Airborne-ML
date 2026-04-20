"""
Generates and saves performance graphs for the Airborne Threat Detection project.
- Training Loss & Accuracy Curve
- FPS Comparison Bar Chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
TRAINING_RESULTS_CSV = Path('train3/results.csv')
OUTPUT_DIR = Path('images')

# Performance data for FPS chart
FPS_DATA = {
    'Method': [
        'YOLO Local (Offline)', 
        'Roboflow API (High Accuracy)', 
        'Roboflow API (Fast - Stride 5)'
    ],
    'FPS': [30, 2, 6]
}

# --- Styling ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def plot_training_curves(df):
    """
    Plots and saves the training loss and mAP accuracy curves.
    
    Args:
        df (pd.DataFrame): DataFrame containing the training results.
    """
    print("📊 Generating Training Loss & Accuracy Curve Graph...")
    
    fig, ax1 = plt.subplots()

    # Plot Training and Validation Loss
    ax1.plot(df['epoch'], df['   train/box_loss'], 'b-', label='Train Box Loss')
    ax1.plot(df['epoch'], df['     val/box_loss'], 'b--', label='Validation Box Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Box Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # Create a second y-axis for mAP
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df["       metrics/mAP50(B)"], 'r-', label='mAP@0.5')
    ax2.set_ylabel('mAP@0.5', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    # Final Touches
    plt.title('Training Loss vs. Accuracy (mAP@0.5)')
    fig.tight_layout()
    
    # Save the figure
    output_path = OUTPUT_DIR / 'training_curves.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved graph to: {output_path}")
    plt.close()


def plot_fps_comparison_chart(data):
    """
    Plots and saves a bar chart comparing the FPS of different detection methods.
    
    Args:
        data (dict): Dictionary with 'Method' and 'FPS' keys.
    """
    print("📊 Generating FPS Comparison Bar Chart...")
    
    df = pd.DataFrame(data)
    
    plt.figure()
    ax = sns.barplot(x='Method', y='FPS', data=df, palette='viridis', hue='Method', dodge=False)

    # Add FPS values on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f FPS', fontsize=12, color='black', padding=3)

    # Final Touches
    ax.set_ylabel('Frames Per Second (FPS)')
    ax.set_xlabel('Detection Method')
    ax.set_ylim(0, max(df['FPS']) * 1.2) # Add some space at the top
    plt.title('Detection Speed: FPS Comparison')
    plt.xticks(rotation=10, ha='right')
    plt.legend([],[], frameon=False) # Hide legend as x-labels are clear
    plt.tight_layout()

    # Save the figure
    output_path = OUTPUT_DIR / 'fps_comparison_chart.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved graph to: {output_path}")
    plt.close()


def main():
    """
    Main function to generate and save all graphs.
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check if training data exists
    if not TRAINING_RESULTS_CSV.exists():
        print(f"❌ Error: Training data not found at '{TRAINING_RESULTS_CSV}'")
        print("Cannot generate training curves graph.")
    else:
        # Clean up column names before plotting
        try:
            training_df = pd.read_csv(TRAINING_RESULTS_CSV)
            training_df.columns = training_df.columns.str.strip()
            plot_training_curves(training_df)
        except Exception as e:
            print(f"❌ Error processing training data: {e}")

    # Generate FPS comparison chart
    plot_fps_comparison_chart(FPS_DATA)
    
    print("\n✨ All graphs generated successfully!")


if __name__ == '__main__':
    main()
