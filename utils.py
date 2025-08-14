import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(
        df: pd.DataFrame,
        metrics = ["UAS", "LAS", "CLAS", "BLEX"]
        ):
    # Create combined label column (Model + Projectivity)
    df["Label"] = df["Model"] + " (" + df["Projective"].map({True: "Proj", False: "Non-Proj"}) + ")"
    
    # Filter metrics to plot
    plot_df = df[df["Metric"].isin(metrics)]
    
    # Set up the plot style for scientific publications
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a more sophisticated color palette
    # Use colorblind-friendly colors with distinct patterns for proj/non-proj
    
    # Plot with improved styling
    sns.barplot(
        data=plot_df, 
        x="F1 Score", 
        y="Metric", 
        hue="Label",
        palette=sns.color_palette("Paired"),
        ax=ax,
        saturation=0.8,
        alpha=0.9
    )
    
    # Customize the plot appearance
    ax.set_xlabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Metric', fontsize=14, fontweight='bold')
    ax.set_title('Dependency Parsing Performance Comparison\nby Metric, Model, and Projectivity', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 100)  # Assuming F1 scores are in percentage
    
    # Format x-axis to show percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    # Customize legend
    legend = ax.legend(
        title='Model Configuration',
        title_fontsize=13,
        fontsize=11,
        loc='lower right',
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=2 if len(plot_df['Label'].unique()) > 4 else 1
    )
    legend.get_title().set_fontweight('bold')
    
    # Add subtle grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels on bars for precise reading
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10, 
                    label_type='edge', padding=3)
    
    # Remove top and right spines for cleaner look
    sns.despine(top=True, right=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Optional: Save the figure for publication
    # plt.savefig('dependency_parsing_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('dependency_parsing_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

def eval_string_to_pandas(data: str) -> pd.DataFrame:
    # Step-by-step conversion
    lines = [line.strip() for line in data.strip().split('\n')]
    lines = [line for line in lines if '+' not in line]  # Remove the separator line
    rows = [line.split('|') for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    # Create DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # Convert numeric columns to floats
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if __name__ == "__main__":
    plot(pd.read_csv("result_ru.csv"))