import csv
import os
import numpy as np

def analyze_frames(dataset_dir):
    filepath = os.path.join(dataset_dir, 'Annot_List.txt')
    if not os.path.exists(filepath):
        print("Annot_List.txt not found.")
        return

    frames = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            if label == 'D0X':
                continue
            try:
                f_count = int(row['frames'])
                frames.append(f_count)
            except ValueError:
                continue
    
    if not frames:
        print("No valid frames found.")
        return

    frames = np.array(frames)
    print(f"Total samples (excluding D0X): {len(frames)}")
    print(f"Min frames: {frames.min()}")
    print(f"Max frames: {frames.max()}")
    print(f"Mean frames: {frames.mean():.2f}")
    print(f"Median frames: {np.median(frames)}")
    print(f"95th percentile: {np.percentile(frames, 95)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    analyze_frames(dataset_dir)
