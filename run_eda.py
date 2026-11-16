import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

def main():
    os.makedirs("figures", exist_ok=True)
    data_dir = os.path.abspath("./data")
    housing = fetch_california_housing(data_home=data_dir, as_frame=True)
    df = housing.frame

    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['MedHouseVal'], kde=True, bins=50, color='royalblue', ax=axes[0])
    axes[0].set_title("Distribution of Median House Value (MedHouseVal)")
    axes[0].set_xlabel("Median House Value ($100k)")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=df['MedHouseVal'], color='skyblue', ax=axes[1])
    axes[1].set_title("Boxplot of Median House Value")
    axes[1].set_xlabel("Median House Value ($100k)")
    plt.tight_layout()
    plt.savefig("figures/target_distribution.png", dpi=200)
    plt.close()
    print("Saved figures/target_distribution.png")

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of California Housing Features", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=200)
    plt.close()
    print("Saved figures/correlation_heatmap.png")

    # 3. MedInc vs MedHouseVal
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.2, color="darkblue")
    plt.title("Median Income vs. Median House Value", fontsize=13)
    plt.xlabel("Median Income ($10k units)")
    plt.ylabel("Median House Value ($100k units)")
    plt.tight_layout()
    plt.savefig("figures/medinc_vs_medhouseval.png", dpi=200)
    plt.close()
    print("Saved figures/medinc_vs_medhouseval.png")

    # 4. Geographic Distribution
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], cmap='viridis', alpha=0.4, s=df['Population']/100)
    plt.colorbar(scatter, label="Median House Value ($100k)")
    plt.title("California Housing Prices by Geographic Location (Bubble size: Population)", fontsize=13)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("figures/geographic_distribution.png", dpi=200)
    plt.close()
    print("Saved figures/geographic_distribution.png")

if __name__ == "__main__":
    main()
