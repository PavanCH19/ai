# # train_model.py
# import pandas as pd
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # -------------------------------
# # 1. Load prepared data
# # -------------------------------
# train_df = pd.read_csv("data_preparation/train.csv")
# val_df   = pd.read_csv("data_preparation/val.csv")

# # -------------------------------
# # 2. Define feature groups
# # -------------------------------
# num_cols  = ["test_score", "total_years", "n_skills", "n_projects"]
# cat_cols  = ["preferred_domain"]
# text_cols = ["skills_text", "projects_text", "titles_text"]
# feature_cols = num_cols + cat_cols + text_cols

# X_train, y_train = train_df[feature_cols], train_df["label"]
# X_val, y_val     = val_df[feature_cols], val_df["label"]

# # -------------------------------
# # 3. Clean features for CatBoost
# # -------------------------------
# # Convert text columns to strings
# for col in text_cols:
#     X_train.loc[:, col] = X_train[col].fillna("").astype(str)
#     X_val.loc[:, col]   = X_val[col].fillna("").astype(str)

# # Convert categorical columns to strings
# for col in cat_cols:
#     X_train.loc[:, col] = X_train[col].fillna("unknown").astype(str)
#     X_val.loc[:, col]   = X_val[col].fillna("unknown").astype(str)

# # -------------------------------
# # 4. Build CatBoost Pools
# # -------------------------------
# cat_idx  = [feature_cols.index(c) for c in cat_cols]
# text_idx = [feature_cols.index(c) for c in text_cols]

# train_pool = Pool(X_train, label=y_train, cat_features=cat_idx, text_features=text_idx)
# val_pool   = Pool(X_val,   label=y_val,   cat_features=cat_idx, text_features=text_idx)

# # -------------------------------
# # 5. Train Model
# # -------------------------------
# model = CatBoostClassifier(
#     loss_function="MultiClass",
#     eval_metric="Accuracy",
#     iterations=600,
#     depth=6,
#     learning_rate=0.08,
#     random_seed=42,
#     verbose=100
# )

# model.fit(train_pool, eval_set=val_pool)

# # -------------------------------
# # 6. Evaluate
# # -------------------------------
# val_pred = model.predict(val_pool)
# print("✅ Accuracy:", accuracy_score(y_val, val_pred)*100, "%")
# print("\n📊 Classification Report:\n", classification_report(y_val, val_pred))
# print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

# # -------------------------------
# # 7. Feature Importance
# # -------------------------------
# importances = model.get_feature_importance(train_pool)
# print("\n🔥 Feature Importance:")
# for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
#     print(f"{col:15s}  {imp:8.3f}")

# # -------------------------------
# # 8. Save Model
# # -------------------------------
# model.save_model("catboost_resume_model.cbm")
# print("💾 Model saved as catboost_resume_model.cbm")



import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_catboost_model(train_path,
                         val_path,
                         model_path="catboost_resume_model_updated.cbm",
                         iterations=600,
                         depth=6,
                         learning_rate=0.08,
                         random_seed=42):
    # -------------------------------
    # 1. Load prepared data
    # -------------------------------
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # -------------------------------
    # 2. Define feature groups
    # -------------------------------
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
    cat_cols  = ["preferred_domain"]
    text_cols = ["skills_text", "projects_text", "titles_text"]
    feature_cols = num_cols + cat_cols + text_cols

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val     = val_df[feature_cols], val_df["label"]

    # -------------------------------
    # 3. Clean features for CatBoost
    # -------------------------------
    for col in text_cols:
        X_train.loc[:, col] = X_train[col].fillna("").astype(str)
        X_val.loc[:, col]   = X_val[col].fillna("").astype(str)

    for col in cat_cols:
        X_train.loc[:, col] = X_train[col].fillna("unknown").astype(str)
        X_val.loc[:, col]   = X_val[col].fillna("unknown").astype(str)

    # -------------------------------
    # 4. Build CatBoost Pools
    # -------------------------------
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]

    train_pool = Pool(X_train, label=y_train, cat_features=cat_idx, text_features=text_idx)
    val_pool   = Pool(X_val,   label=y_val,   cat_features=cat_idx, text_features=text_idx)

    # -------------------------------
    # 5. Train Model
    # -------------------------------
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        verbose=100
    )

    model.fit(train_pool, eval_set=val_pool)

    # -------------------------------
    # 6. Evaluate
    # -------------------------------
    val_pred = model.predict(val_pool)
    print("✅ Accuracy:", accuracy_score(y_val, val_pred)*100, "%")
    print("\n📊 Classification Report:\n", classification_report(y_val, val_pred))
    print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

    # -------------------------------
    # 7. Feature Importance
    # -------------------------------
    importances = model.get_feature_importance(train_pool)
    print("\n🔥 Feature Importance:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"{col:25s}  {imp:8.3f}")

    # -------------------------------
    # 8. Save Model
    # -------------------------------
    model.save_model(model_path)
    print(f"💾 Model saved as {model_path}")

    return model




