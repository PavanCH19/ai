# prepare_dataset_updated.py
import pandas as pd
import re
import time  # For timing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def prepare_dataset(dataset_path, train_output, val_output):
    start_time = time.time()
    print("⏳ Starting dataset preparation...")

    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    t0 = time.time()
    df = pd.read_csv(dataset_path)
    t1 = time.time()
    print(f"📂 Loaded dataset ({len(df)} samples) in {t1 - t0:.4f} seconds.")

    # -------------------------------
    # 2. Basic cleaning
    # -------------------------------
    t_clean_start = time.time()
    df = df.drop_duplicates()

    # Keep only valid labels
    valid_labels = ["fit", "partial", "suggest", "no_fit"]
    df = df[df["label"].isin(valid_labels)]

    # Fill missing values
    df = df.fillna({
        "test_score": 0,
        "skills_text": "",
        "projects_text": "",
        "titles_text": "",
        "preferred_domain": "",
        "skill_match_ratio": 0,
        "relevant_experience": 0,
        "project_match": 0
    })

    # Clip numeric columns
    df["test_score"] = df["test_score"].clip(0, 100)
    df["skill_match_ratio"] = df["skill_match_ratio"].clip(0, 1)
    df["relevant_experience"] = df["relevant_experience"].clip(lower=0)
    df["project_match"] = df["project_match"].clip(0, 1)
    t_clean_end = time.time()
    print(f"🧹 Basic cleaning completed in {t_clean_end - t_clean_start:.4f} seconds.")

    # -------------------------------
    # 3. Text cleaning
    # -------------------------------
    t_text_start = time.time()
    def clean_text(s):
        s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
        return s.strip()

    for col in ["skills_text", "projects_text", "titles_text", "preferred_domain"]:
        df[col] = df[col].apply(clean_text)
    t_text_end = time.time()
    print(f"📝 Text cleaning completed in {t_text_end - t_text_start:.4f} seconds.")

    # -------------------------------
    # 4. Oversampling to balance labels
    # -------------------------------
    t_balance_start = time.time()
    X = df.drop(columns=["label"])
    y = df["label"]

    print("Before balancing:", y.value_counts().to_dict())

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    balanced_df = pd.concat([X_res, y_res], axis=1).reset_index(drop=True)
    print("After balancing:", balanced_df['label'].value_counts().to_dict())
    t_balance_end = time.time()
    print(f"⚖️ Oversampling completed in {t_balance_end - t_balance_start:.4f} seconds.")

    # -------------------------------
    # 5. Train/Validation split
    # -------------------------------
    t_split_start = time.time()
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        random_state=42,
        stratify=balanced_df["label"],
        shuffle=True
    )
    t_split_end = time.time()
    print(f"🔹 Train/Validation split completed in {t_split_end - t_split_start:.4f} seconds.")

    # Save datasets
    t_save_start = time.time()
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    t_save_end = time.time()
    print(f"💾 Train/Validation datasets saved in {t_save_end - t_save_start:.4f} seconds.")

    end_time = time.time()
    print(f"✅ Dataset preparation finished in {end_time - start_time:.2f} seconds.")
    print("Train size:", len(train_df), "Validation size:", len(val_df))

    return train_df, val_df
