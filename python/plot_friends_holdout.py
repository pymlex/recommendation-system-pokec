import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/friends_holdout_results.csv"
    if not os.path.exists(path):
        print("File not found:", path)
        return
    df = pd.read_csv(path, header=None, names=['ratio'])
    plt.figure(figsize=(8,6))
    df['ratio'] = df[df['ratio'] > 0] * 4
    sns.histplot(df['ratio'], kde=True, bins=10)
    plt.xlabel('Correct prediction ratio')
    plt.ylabel('Count')
    plt.title('Friends holdout: distribution of per-user correct-prediction ratio')
    plt.tight_layout()
    out = os.path.splitext(path)[0] + "_hist_kde.png"
    plt.savefig(out)
    print("Saved", out)

if __name__ == '__main__':
    main()