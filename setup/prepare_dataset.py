# # prepare_dataset.py
# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler

# path = r"C:\Users\pavan\Desktop\interview-ai\data_generation\candidates_dataset.csv"

# # Load dataset (CSV instead of JSON)
# df = pd.read_csv(path)

# # Cleaning
# df = df.drop_duplicates()
# valid_labels = ["fit", "partial", "suggest", "no_fit"]
# df = df[df["label"].isin(valid_labels)]
# df = df.fillna({"test_score": 0, "skills_text": "", "projects_text": "", "titles_text": ""})
# df["test_score"] = df["test_score"].clip(0, 100)

# def clean_text(s):
#     s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
#     return s.strip()

# for col in ["skills_text", "projects_text", "titles_text"]:
#     df[col] = df[col].apply(clean_text)

# # Oversampling to balance labels
# X = df.drop(columns=["label"])
# y = df["label"]

# print("Before balancing:", y.value_counts().to_dict())

# ros = RandomOverSampler(random_state=42)
# X_res, y_res = ros.fit_resample(X, y)

# balanced_df = pd.concat([X_res, y_res], axis=1).reset_index(drop=True)

# print("After balancing:", balanced_df['label'].value_counts().to_dict())

# # Train/Validation split
# train_df, val_df = train_test_split(
#     balanced_df, test_size=0.2, random_state=42, stratify=balanced_df["label"]
# )

# train_df.to_csv("train.csv", index=False)
# val_df.to_csv("val.csv", index=False)

# print("✅ Dataset prepared with oversampling: train.csv & val.csv")
# print("Train size:", len(train_df), "Validation size:", len(val_df))


# prepare_dataset_updated.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def prepare_dataset(dataset_path, train_output, val_output):
    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    df = pd.read_csv(dataset_path)

    # -------------------------------
    # 2. Basic cleaning
    # -------------------------------
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

    # -------------------------------
    # 3. Text cleaning
    # -------------------------------
    def clean_text(s):
        s = re.sub(r"[^a-z0-9\s]", "", str(s).lower())
        return s.strip()

    for col in ["skills_text", "projects_text", "titles_text", "preferred_domain"]:
        df[col] = df[col].apply(clean_text)

    # -------------------------------
    # 4. Oversampling to balance labels
    # -------------------------------
    X = df.drop(columns=["label"])
    y = df["label"]

    print("Before balancing:", y.value_counts().to_dict())

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    balanced_df = pd.concat([X_res, y_res], axis=1).reset_index(drop=True)

    print("After balancing:", balanced_df['label'].value_counts().to_dict())

    # -------------------------------
    # 5. Train/Validation split
    # -------------------------------
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        random_state=42,
        stratify=balanced_df["label"],
        shuffle=True
    )

    # Save datasets
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"✅ Dataset prepared: {train_output} & {val_output}")
    print("Train size:", len(train_df), "Validation size:", len(val_df))

    return train_df, val_df

