from dfGeneration import update_dataset
# from prepare_dataset import prepare_dataset

# train_df, val_df = prepare_dataset(dataset_path=r"C:\Users\pavan\Desktop\interview-ai\output\candidates_pred.csv", 
#                                    train_output=r"C:\Users\pavan\Desktop\interview-ai\output\train_data.csv", 
#                                    val_output=r"C:\Users\pavan\Desktop\interview-ai\output\val_data.csv")

import pandas as pd
from catboost import CatBoostClassifier, Pool

def predict_candidates(model_path, input_csv, output_csv="predictions.csv"):
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    # Load new data
    df = pd.read_csv(input_csv)

    # Feature groups
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
    cat_cols  = ["preferred_domain"]
    text_cols = ["skills_text", "projects_text", "titles_text"]
    feature_cols = num_cols + cat_cols + text_cols

    X = df[feature_cols].copy()  # <-- copy to avoid SettingWithCopyWarning

    # Clean features
    for col in text_cols:
        X[col] = X[col].fillna("").astype(str)
    for col in cat_cols:
        X[col] = X[col].fillna("unknown").astype(str)

    # Build Pool
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]
    pool = Pool(X, cat_features=cat_idx, text_features=text_idx)

    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Predict
    predictions = model.predict(pool).ravel()  # <-- flatten 2D array to 1D
    df["predicted_label"] = predictions

    # Display predictions
    print("\n📌 Predictions:")
    print(df[["test_score", "skill_match_ratio", "relevant_experience", "project_match",
              "preferred_domain", "predicted_label"]])

    # Save predictions
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")

    return df
# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    update_dataset(
        dataset_path="candidate_prediction.json",
        domain_path="domain_requirements.json",
        output_csv=r"C:\Users\pavan\Desktop\interview-ai\output\candidates_pred.csv",
        output_parquet=r"C:\Users\pavan\Desktop\interview-ai\output\candidates_pred.parquet"
    )
    input_csv = r"C:\Users\pavan\Desktop\interview-ai\output\candidates_pred.csv"  # your CSV with new data
    model_path = "catboost_resume_model_updated.cbm"
    predict_candidates(model_path, input_csv, output_csv="final_output.csv")
    
