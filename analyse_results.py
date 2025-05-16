import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def bucket_label(length):
    """Assign each length to a predefined bucket."""
    if length == 1:
        return '1'
    elif length == 2:
        return '2'
    elif length == 3:
        return '3'
    elif 4 <= length <= 10:
        return '4-10'
    elif 11 <= length <= 100:
        return '11-100'
    else:
        return '100+'


def analyze_file(filepath):
    """
    Load a .jsonl file, compute response-length statistics (bucketed),
    and display a log-scaled histogram + boxplot for easy interpretation.
    """
    # Read all JSON lines
    responses = []
    with open(filepath, 'r') as f:
        for line in f:
            responses.append(json.loads(line))

    # Print header and summary
    print(f"Analyzing file: {filepath}")
    print("=" * 80)
    print(f"Total responses: {len(responses)}")

    # Compute raw response lengths and bucket them
    responses = [res["resps"][0][0].strip() for res in responses]
    raw_lengths = [len(res) for res in responses]

    for i in range(10):
        print(responses[i])
    bucketed = [bucket_label(l) for l in raw_lengths]
    count_map = Counter(bucketed)

    # Print bucketed counts
    print("Response length buckets:")
    for bucket in ['1', '2', '3', '4-10', '11-100', '100+']:
        print(f"  {bucket}: {count_map.get(bucket, 0)}")

    # # Convert to numpy array for plotting
    # lengths = np.array(raw_lengths)
    #
    # # Plot 1: Histogram with log-scaled Y-axis
    # plt.figure(figsize=(10, 4))
    # plt.hist(lengths, bins=[1,2,3,4,11,101, lengths.max()+1], edgecolor='black')
    # plt.yscale('log')
    # plt.title(f"{os.path.basename(filepath)} — Response Length Distribution (log scale)")
    # plt.xlabel("Response Length Buckets")
    # plt.ylabel("Frequency (log scale)")
    # plt.xticks([1.5,2.5,3.5,7,55.5, (lengths.max()+1+101)/2], ['1','2','3','4-10','11-100','100+'])
    # plt.tight_layout()
    # plt.show()
    #
    # # Plot 2: Horizontal boxplot
    # plt.figure(figsize=(6, 4))
    # plt.boxplot(lengths, vert=False)
    # plt.title(f"{os.path.basename(filepath)} — Response Length Boxplot")
    # plt.xlabel("Response Length")
    # plt.tight_layout()
    # plt.show()
    #
    # print("\n")


if __name__ == '__main__':
    # Root directory to scan
    root_dir = "/data/users/pjajoria/ICL_runs"

    # Walk through all subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            # Match files starting with samples_bbbp_llama3 and ending in .jsonl
            if fname.startswith("samples_bbbp_gemma2-27b") and fname.endswith(".jsonl"):
                full_path = os.path.join(dirpath, fname)
                analyze_file(full_path)
