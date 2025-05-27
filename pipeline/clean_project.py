import os

# Define paths and what to clean
paths_to_clean = {
    "../data/processed": [".csv", ".npz"],
    "../models": [".pkl", ".joblib"],
    "../summaries": [".csv"],
    "../figures/eda": [".png"],
    "../figures/evaluation": [".png"],
    "../data/input": [".csv"]
}

# Function to safely delete only matching files
def clean():
    for path, extensions in paths_to_clean.items():
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and any(file.endswith(ext) for ext in extensions):
                os.remove(file_path)
                print(f"🗑️ Removed: {file_path}")

if __name__ == "__main__":
    print("🚿 Cleaning intermediate files...")
    clean()
    print("✅ Done.")

