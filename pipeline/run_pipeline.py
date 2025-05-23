import subprocess

steps = [
    "load_clean.py",
    "summarize_data.py",
    "preprocess_model_data.py",
    "train_models.py",
    "evaluate_models.py",
    "feature_importance.py",
    "generate_figures.py"
]

print("🚀 Running full NHANES ML pipeline...")

for script in steps:
    print(f"\n▶ Running: {script}")
    subprocess.run(["python", script], check=True)

print("\n✅ Pipeline complete. All outputs available in data/, models/, summaries/, figures/.")

