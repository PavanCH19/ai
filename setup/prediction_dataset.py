from pathlib import Path
import time
from dfGeneration import update_dataset
import pandas as pd
from catboost import CatBoostClassifier, Pool

def update_and_predict(dataset_path, domain_path, model_path, output_dir):
    """
    Updates the dataset and runs predictions.
    
    Args:
        dataset_path (str or Path): Path to candidates JSON.
        domain_path (str or Path): Path to domain requirements JSON.
        model_path (str or Path): Path to CatBoost model.
        output_dir (str or Path): Directory to save outputs.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_csv = output_dir / "candidates_pred.csv"
    output_parquet = output_dir / "candidates_pred.parquet"
    final_output_csv = output_dir / "final_output.csv"

    # ---------------------------
    # Step 1: Update Dataset
    # ---------------------------
    print("🔹 Updating dataset...")
    t_update_start = time.time()
    update_dataset(
        dataset_path=str(dataset_path),
        domain_path=str(domain_path),
        output_csv=str(output_csv),
        output_parquet=str(output_parquet)
    )
    t_update_end = time.time()
    print(f"⏱ Dataset update completed in {t_update_end - t_update_start:.2f} seconds.\n")

    # ---------------------------
    # Step 2: Run Predictions
    # ---------------------------
    print("🔹 Running predictions...")
    start_time = time.time()

    # Load new data
    df = pd.read_csv(output_csv)
    print(f"📂 Loaded input CSV ({len(df)} samples)")

    # Feature groups
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
    cat_cols  = ["preferred_domain"]
    text_cols = ["skills_text", "projects_text", "titles_text"]
    feature_cols = num_cols + cat_cols + text_cols

    X = df[feature_cols].copy()

    # Clean features
    for col in text_cols:
        X[col] = X[col].fillna("").astype(str)
    for col in cat_cols:
        X[col] = X[col].fillna("unknown").astype(str)

    # Build CatBoost Pool
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]
    pool = Pool(X, cat_features=cat_idx, text_features=text_idx)

    # Load model
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Predict
    df["predicted_label"] = model.predict(pool).ravel()

    # Show top 10 predictions
    print("\n📌 Predictions:")
    print(df[["test_score", "skill_match_ratio", "relevant_experience", "project_match",
              "preferred_domain", "predicted_label"]].head(10))

    # Save predictions
    df.to_csv(final_output_csv, index=False)
    df.to_parquet(final_output_csv.with_suffix(".parquet"), index=False)

    end_time = time.time()
    print(f"✅ Total prediction process finished in {end_time - start_time:.2f} seconds.\n")

    return df


# ---------------------------
# Example usage from any file
# ---------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    dataset_path = BASE_DIR / "candidates_prediction.json"
    domain_path = BASE_DIR / "domain_requirements.json"
    model_path = BASE_DIR / "catboost_resume_model_updated.cbm"
    output_dir = BASE_DIR / "output"

    update_and_predict(dataset_path, domain_path, model_path, output_dir)





# from dfGeneration import update_dataset
# from pathlib import Path
# import time
# import pandas as pd
# from catboost import CatBoostClassifier, Pool

# def predict_candidates_updated(model_path, input_csv, output_csv="predictions.csv"):
#     start_time = time.time()
#     print("⏳ Starting prediction process...")

#     # Load new data
#     df = pd.read_csv(input_csv)
#     print(f"📂 Loaded input CSV ({len(df)} samples).")

#     # Feature groups
#     num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
#     cat_cols  = ["preferred_domain"]
#     text_cols = ["skills_text", "projects_text", "titles_text"]
#     feature_cols = num_cols + cat_cols + text_cols

#     X = df[feature_cols].copy()

#     # Clean features
#     for col in text_cols:
#         X[col] = X[col].fillna("").astype(str)
#     for col in cat_cols:
#         X[col] = X[col].fillna("unknown").astype(str)

#     # Build Pool
#     cat_idx  = [feature_cols.index(c) for c in cat_cols]
#     text_idx = [feature_cols.index(c) for c in text_cols]
#     pool = Pool(X, cat_features=cat_idx, text_features=text_idx)

#     # Load model
#     model = CatBoostClassifier()
#     model.load_model(model_path)

#     # Predict
#     predictions = model.predict(pool).ravel()
#     df["predicted_label"] = predictions

#     # --- Adjust predictions based on thresholds ---
#     def adjust_label(row):
#         if row["test_score"] >= 80 and row["skill_match_ratio"] > 0.35 and row["relevant_experience"] > 1:
#             return "fit"
#         elif row["test_score"] >= 70 and row["skill_match_ratio"] > 0.25 and row["relevant_experience"] >= 1:
#             return "partial"
#         else:
#             return row["predicted_label"]  # keep CatBoost suggestion for no_fit/suggest

#     df["predicted_label"] = df.apply(adjust_label, axis=1)

#     # Display top predictions
#     print("\n📌 Predictions:")
#     print(df[["test_score", "skill_match_ratio", "relevant_experience", "project_match",
#               "preferred_domain", "predicted_label"]].head(10))

#     # Save predictions
#     df.to_csv(output_csv, index=False)
#     print(f"💾 Predictions saved to {output_csv}")

#     print(f"✅ Total prediction process finished in {time.time() - start_time:.2f} seconds.")
#     return df

# # -------------------------------
# # Example usage
# # -------------------------------
# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).parent.resolve()
#     OUTPUT_DIR = BASE_DIR / "output"
#     OUTPUT_DIR.mkdir(exist_ok=True)

#     dataset_path = BASE_DIR / "candidates_prediction.json"
#     domain_path = BASE_DIR / "domain_requirements.json"
#     output_csv = OUTPUT_DIR / "candidates_pred.csv"
#     output_parquet = OUTPUT_DIR / "candidates_pred.parquet"

#     print("🔹 Updating dataset...")
#     update_dataset(
#         dataset_path=str(dataset_path),
#         domain_path=str(domain_path),
#         output_csv=str(output_csv),
#         output_parquet=str(output_parquet)
#     )

#     input_csv = output_csv
#     model_path = BASE_DIR / "catboost_resume_model_updated.cbm"
#     final_output_csv = OUTPUT_DIR / "final_output.csv"

#     print("🔹 Running predictions...")
#     predict_candidates_updated(
#         model_path=str(model_path),
#         input_csv=str(input_csv),
#         output_csv=str(final_output_csv)
#     )
