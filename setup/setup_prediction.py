from prediction_dataset import update_and_predict
from explain_predictions import generate_explanations
from pathlib import Path

BASE_DIR = Path("C:/Users/pavan/Desktop/ai/setup")

update_and_predict(
    dataset_path=BASE_DIR / "candidates_prediction.json",
    domain_path=BASE_DIR / "domain_requirements.json",
    model_path=BASE_DIR / "catboost_resume_model_updated.cbm",
    output_dir=BASE_DIR / "output"
)



results = generate_explanations(
    candidates_json=BASE_DIR / "candidates_prediction.json",
    domain_json=BASE_DIR / "domain_requirements.json",
    predictions_csv=BASE_DIR / "output/final_output.csv",
    output_json=BASE_DIR / "output/explanations.json"
)
