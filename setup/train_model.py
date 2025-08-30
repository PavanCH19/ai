# import pandas as pd
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import time

# def train_catboost_model(train_path,
#                          val_path,
#                          model_path="catboost_resume_model_updated.cbm",
#                          iterations=600,
#                          depth=6,
#                          learning_rate=0.08,
#                          random_seed=42):
#     start_time = time.time()
#     print("⏳ Starting CatBoost model training...\n")

#     # -------------------------------
#     # Step 1: Load prepared data
#     # -------------------------------
#     t0 = time.time()
#     print("Step 1/8: Loading train and validation datasets...")
#     train_df = pd.read_csv(train_path)
#     val_df   = pd.read_csv(val_path)
#     t1 = time.time()
#     print(f"✅ Loaded {len(train_df)} training rows and {len(val_df)} validation rows in {t1 - t0:.2f} seconds.\n")

#     # -------------------------------
#     # Step 2: Define feature groups
#     # -------------------------------
#     t2 = time.time()
#     print("Step 2/8: Defining feature groups...")
#     num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
#     cat_cols  = ["preferred_domain"]
#     text_cols = ["projects_text", "titles_text"]  
#     feature_cols = num_cols + cat_cols + text_cols
#     print(f"Features: {feature_cols}")
#     X_train, y_train = train_df[feature_cols].copy(), train_df["label"]
#     X_val, y_val     = val_df[feature_cols].copy(), val_df["label"]
#     t3 = time.time()
#     print(f"✅ Feature groups defined in {t3 - t2:.2f} seconds.\n")

#     # -------------------------------
#     # Step 3: Clean features
#     # -------------------------------
#     t4 = time.time()
#     print("Step 3/8: Cleaning text and categorical features...")
#     for col in text_cols:
#         X_train.loc[:, col] = X_train[col].fillna("").astype(str)
#         X_val.loc[:, col]   = X_val[col].fillna("").astype(str)
#     for col in cat_cols:
#         X_train.loc[:, col] = X_train[col].fillna("unknown").astype(str)
#         X_val.loc[:, col]   = X_val[col].fillna("unknown").astype(str)
#     t5 = time.time()
#     print(f"✅ Features cleaned in {t5 - t4:.2f} seconds.\n")

#     # -------------------------------
#     # Step 4: Build CatBoost Pools
#     # -------------------------------
#     t6 = time.time()
#     print("Step 4/8: Building CatBoost Pools...")
#     cat_idx  = [feature_cols.index(c) for c in cat_cols]
#     text_idx = [feature_cols.index(c) for c in text_cols]
#     train_pool = Pool(X_train, label=y_train, cat_features=cat_idx, text_features=text_idx)
#     val_pool   = Pool(X_val,   label=y_val,   cat_features=cat_idx, text_features=text_idx)
#     t7 = time.time()
#     print(f"✅ Pools created in {t7 - t6:.2f} seconds.\n")

#     # -------------------------------
#     # Step 5: Train Model
#     # -------------------------------
#     t8 = time.time()
#     print("Step 5/8: Training CatBoost model...")
#     model = CatBoostClassifier(
#         loss_function="MultiClass",
#         eval_metric="Accuracy",
#         iterations=iterations,
#         depth=depth,
#         learning_rate=learning_rate,
#         random_seed=random_seed,
#         verbose=100
#     )
#     model.fit(train_pool, eval_set=val_pool)
#     t9 = time.time()
#     print(f"✅ Model training completed in {t9 - t8:.2f} seconds.\n")

#     # -------------------------------
#     # Step 6: Evaluate
#     # -------------------------------
#     t10 = time.time()
#     print("Step 6/8: Evaluating model...")
#     val_pred = model.predict(val_pool)
#     acc = accuracy_score(y_val, val_pred)*100
#     print(f"✅ Accuracy: {acc:.2f}%")
#     print("\n📊 Classification Report:\n", classification_report(y_val, val_pred))
#     print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_val, val_pred))
#     t11 = time.time()
#     print(f"✅ Evaluation completed in {t11 - t10:.2f} seconds.\n")

#     # -------------------------------
#     # Step 7: Feature Importance
#     # -------------------------------
#     t12 = time.time()
#     print("Step 7/8: Feature importance:")
#     importances = model.get_feature_importance(train_pool)
#     for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
#         print(f"{col:25s}  {imp:8.3f}")
#     t13 = time.time()
#     print(f"✅ Feature importance computed in {t13 - t12:.2f} seconds.\n")

#     # -------------------------------
#     # Step 8: Save Model
#     # -------------------------------
#     t14 = time.time()
#     model.save_model(model_path)
#     t15 = time.time()
#     print(f"💾 Model saved as {model_path} in {t15 - t14:.2f} seconds.\n")

#     end_time = time.time()
#     print(f"⏱ Total training process took {end_time - start_time:.2f} seconds")

#     return model



import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def train_catboost_model(train_path,
                         val_path,
                         model_path="catboost_resume_model_updated.cbm",
                         iterations=2000,
                         depth=8,
                         learning_rate=0.05,
                         random_seed=42):
    start_time = time.time()
    print("⏳ Starting CatBoost model training...\n")

    # -------------------------------
    # Step 1: Load prepared data
    # -------------------------------
    print("Step 1/8: Loading train and validation datasets...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    print(f"✅ Loaded {len(train_df)} training rows and {len(val_df)} validation rows.\n")

    # -------------------------------
    # Step 2: Define feature groups
    # -------------------------------
    print("Step 2/8: Defining feature groups...")
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]
    cat_cols  = ["preferred_domain"]
    text_cols = ["skills_text", "projects_text", "titles_text"]
    feature_cols = num_cols + cat_cols + text_cols
    print(f"Features: {feature_cols}")

    X_train, y_train = train_df[feature_cols].copy(), train_df["label"]
    X_val, y_val     = val_df[feature_cols].copy(), val_df["label"]

    # -------------------------------
    # Step 3: Clean features
    # -------------------------------
    print("Step 3/8: Cleaning text and categorical features...")
    for col in text_cols:
        X_train[col] = X_train[col].fillna("").astype(str)
        X_val[col]   = X_val[col].fillna("").astype(str)
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("unknown").astype(str)
        X_val[col]   = X_val[col].fillna("unknown").astype(str)

    # -------------------------------
    # Step 4: Build CatBoost Pools
    # -------------------------------
    print("Step 4/8: Building CatBoost Pools...")
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_idx, text_features=text_idx)
    val_pool   = Pool(X_val, label=y_val, cat_features=cat_idx, text_features=text_idx)

    # -------------------------------
    # Step 5: Train Model
    # -------------------------------
    print("Step 5/8: Training CatBoost model...")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        od_type="Iter",
        od_wait=50,
        l2_leaf_reg=5,
        random_strength=0.5,
        bagging_temperature=1,
        auto_class_weights="Balanced",
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool)
    print("✅ Model training completed.\n")

    # -------------------------------
    # Step 6: Evaluate
    # -------------------------------
    print("Step 6/8: Evaluating model...")
    val_pred = model.predict(val_pool)
    acc = accuracy_score(y_val, val_pred) * 100
    print(f"✅ Accuracy: {acc:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_val, val_pred))
    print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

    # -------------------------------
    # Step 7: Feature Importance
    # -------------------------------
    print("Step 7/8: Feature importance:")
    importances = model.get_feature_importance(train_pool)
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"{col:25s}  {imp:8.3f}")

    # -------------------------------
    # Step 8: Save Model
    # -------------------------------
    model.save_model(model_path)
    print(f"💾 Model saved as {model_path}\n")

    print(f"⏱ Total training process took {time.time() - start_time:.2f} seconds")
    return model
