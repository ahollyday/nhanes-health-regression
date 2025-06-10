import optuna
import optuna.visualization as vis
import os

models = ["random_forest", "gradient_boosting", "knn", "svr", "xgboost"]
study_dir = "../optuna_studies"
output_dir = "../figures/evaluation/optuna_visuals"
os.makedirs(output_dir, exist_ok=True)

for model in models:
    print(f" Generating visualizations for: {model}")
    study = optuna.load_study(
        study_name=model,
        storage=f"sqlite:///{os.path.join(study_dir, f'optuna_{model}.db')}"
    )

    vis.plot_optimization_history(study).write_image(f"{output_dir}/{model}_opt_history.png")
    vis.plot_param_importances(study).write_image(f"{output_dir}/{model}_param_importance.png")
    vis.plot_parallel_coordinate(study).write_image(f"{output_dir}/{model}_parallel_coord.png")

print(" All PNG plots saved to:", output_dir)

