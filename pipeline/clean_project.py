import os

# Define paths and extensions to clean
paths_to_clean = {
    "../data/processed": [".csv", ".npz"],
    "../models": [".pkl", ".joblib"],
    "../summaries": [".csv"],
    "../figures/eda": [".png"],
    "../figures/evaluation": [".png"],
    "../figures/evaluation/optuna_visuals": [".png"],
    "../data/input": [".csv"],
    "../optuna_studies": [".db"],
    "../data/predict_processed": [".npy", ".csv"],
    "../figures/predictions/": [".png"],
    "../figures/predictions/eda": [".png"],    
    "../figures/feature_importance": [".png"]
}

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

