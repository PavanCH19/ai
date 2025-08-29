from dataGeneration import generate_dataset
from dfGeneration import update_dataset
from prepare_dataset import prepare_dataset
from train_model import train_catboost_model

generate_dataset(domain_path="domain_requirements.json", total_samples=5000, output_path=r"C:\Users\pavan\Desktop\interview-ai\output\custom_dataset.json")

update_dataset(
    dataset_path=r"C:\Users\pavan\Desktop\interview-ai\output\custom_dataset.json",
    domain_path="domain_requirements.json",
    output_csv=r"C:\Users\pavan\Desktop\interview-ai\output\candidates.csv",
    output_parquet=r"C:\Users\pavan\Desktop\interview-ai\output\candidates.parquet"
)

train_df, val_df = prepare_dataset(dataset_path=r"C:\Users\pavan\Desktop\interview-ai\output\candidates.csv", 
                                   train_output=r"C:\Users\pavan\Desktop\interview-ai\output\train_data.csv", 
                                   val_output=r"C:\Users\pavan\Desktop\interview-ai\output\val_data.csv")

model = train_catboost_model(
    train_path=r"C:\Users\pavan\Desktop\interview-ai\output\train_data.csv",
    val_path=r"C:\Users\pavan\Desktop\interview-ai\output\val_data.csv",
    model_path="catboost_resume_model_updated.cbm"
)
