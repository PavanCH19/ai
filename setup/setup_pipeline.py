from pathlib import Path
from dataGeneration import generate_dataset
from dfGeneration import update_dataset
from prepare_dataset import prepare_dataset
from train_model import train_catboost_model
import time

# Base directory of this script
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)  # create output folder if it doesn't exist

# Paths
domain_path = BASE_DIR / "domain_requirements.json"
custom_dataset_path = OUTPUT_DIR / "custom_dataset.json"
candidates_csv = OUTPUT_DIR / "candidates.csv"
candidates_parquet = OUTPUT_DIR / "candidates.parquet"
train_csv = OUTPUT_DIR / "train_data.csv"
val_csv = OUTPUT_DIR / "val_data.csv"
model_path = BASE_DIR / "catboost_resume_model_updated.cbm"

# -------------------------------
# Step 1: Generate dataset
# -------------------------------
print("Step 1/4: Generating custom dataset...")
t1 = time.time()
generate_dataset(domain_path="domain_requirements.json", output_path=custom_dataset_path, total_samples=10000)
t2 = time.time()
print(f"✅ Custom dataset generated and saved to: {custom_dataset_path} in {t2 - t1:.2f} seconds\n")

# -------------------------------
# Step 2: Update dataset
# -------------------------------
print("Step 2/4: Updating dataset with features...")
t3 = time.time()
update_dataset(
    dataset_path=str(custom_dataset_path),
    domain_path=str(domain_path),
    output_csv=str(candidates_csv),
    output_parquet=str(candidates_parquet)
)
t4 = time.time()
print(f"✅ Updated dataset saved to CSV: {candidates_csv}")
print(f"✅ Updated dataset saved to Parquet: {candidates_parquet}")
print(f"⏱ Step 2 completed in {t4 - t3:.2f} seconds\n")

# -------------------------------
# Step 3: Prepare train and validation sets
# -------------------------------
print("Step 3/4: Preparing train and validation datasets...")
t5 = time.time()
train_df, val_df = prepare_dataset(
    dataset_path=str(candidates_csv),
    train_output=str(train_csv),
    val_output=str(val_csv)
)
t6 = time.time()
print(f"✅ Training data saved to: {train_csv}")
print(f"✅ Validation data saved to: {val_csv}")
print(f"⏱ Step 3 completed in {t6 - t5:.2f} seconds\n")

# -------------------------------
# Step 4: Train CatBoost model
# -------------------------------
print("Step 4/4: Training CatBoost model...")
t7 = time.time()
model = train_catboost_model(
    train_path=str(train_csv),
    val_path=str(val_csv),
    model_path=str(model_path)
)
t8 = time.time()
print(f"✅ Model training completed. Model saved to: {model_path}")
print(f"⏱ Step 4 completed in {t8 - t7:.2f} seconds\n")

# -------------------------------
# Total pipeline time
# -------------------------------
total_time = t8 - t1
print(f"🏁 Total pipeline completed in {total_time:.2f} seconds")
