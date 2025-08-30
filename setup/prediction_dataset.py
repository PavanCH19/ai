from dfGeneration import update_dataset
from pathlib import Path
import time  # For timing measurements

def predict_candidates(model_path, input_csv, output_csv="predictions.csv"):
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    start_time = time.time()
    print("⏳ Starting prediction process...")

    # Load new data
    t0 = time.time()
    df = pd.read_csv(input_csv)
    t1 = time.time()
    print(f"📂 Loaded input CSV ({len(df)} samples) in {t1 - t0:.4f} seconds.")

    # Feature groups
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
    cat_cols  = ["preferred_domain"]
    text_cols = ["skills_text", "projects_text", "titles_text"]
    feature_cols = num_cols + cat_cols + text_cols

    X = df[feature_cols].copy()  # <-- copy to avoid SettingWithCopyWarning

    # Clean features
    t_clean_start = time.time()
    for col in text_cols:
        X[col] = X[col].fillna("").astype(str)
    for col in cat_cols:
        X[col] = X[col].fillna("unknown").astype(str)
    t_clean_end = time.time()
    print(f"🧹 Features cleaned in {t_clean_end - t_clean_start:.4f} seconds.")

    # Build Pool
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]
    t_pool_start = time.time()
    pool = Pool(X, cat_features=cat_idx, text_features=text_idx)
    t_pool_end = time.time()
    print(f"🧩 CatBoost Pool built in {t_pool_end - t_pool_start:.4f} seconds.")

    # Load model
    t_model_start = time.time()
    model = CatBoostClassifier()
    model.load_model(model_path)
    t_model_end = time.time()
    print(f"📦 Model loaded in {t_model_end - t_model_start:.4f} seconds.")

    # Predict
    t_pred_start = time.time()
    predictions = model.predict(pool).ravel()  # <-- flatten 2D array to 1D
    df["predicted_label"] = predictions
    t_pred_end = time.time()
    print(f"⚡ Predictions made in {t_pred_end - t_pred_start:.4f} seconds.")

    # Display predictions
    print("\n📌 Predictions:")
    print(df[["test_score", "skill_match_ratio", "relevant_experience", "project_match",
              "preferred_domain", "predicted_label"]].head(10))  # show top 10 for brevity

    # Save predictions
    t_save_start = time.time()
    df.to_csv(output_csv, index=False)
    t_save_end = time.time()
    print(f"💾 Predictions saved to {output_csv} in {t_save_end - t_save_start:.4f} seconds.")

    end_time = time.time()
    print(f"✅ Total prediction process finished in {end_time - start_time:.2f} seconds.")

    return df

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()  # directory of the current script
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)  # create output folder if it doesn't exist

    dataset_path = BASE_DIR / "candidates_prediction.json"
    domain_path = BASE_DIR / "domain_requirements.json"
    output_csv = OUTPUT_DIR / "candidates_pred.csv"
    output_parquet = OUTPUT_DIR / "candidates_pred.parquet"

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

    input_csv = output_csv  # your CSV with new data
    model_path = BASE_DIR / "catboost_resume_model_updated.cbm"
    final_output_csv = OUTPUT_DIR / "final_output.csv"

    print("🔹 Running predictions...")
    predict_candidates(
        model_path=str(model_path),
        input_csv=str(input_csv),
        output_csv=str(final_output_csv)
    )




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
