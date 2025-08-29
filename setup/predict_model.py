import pandas as pd
from catboost import CatBoostClassifier, Pool
import json

DOMAIN_PATH = r"C:\Users\pavan\Desktop\interview-ai\data_generation\domain_requirements.json"

# Load domain requirements
with open(DOMAIN_PATH, "r", encoding="utf-8") as f:
    DOMAIN_REQUIREMENTS = json.load(f)

def json_to_df(json_data):
    """Convert JSON candidate(s) to DataFrame with computed features."""
    if isinstance(json_data, dict):
        json_data = [json_data]

    records = []
    for cand in json_data:
        skills = set(cand.get("skills", []))
        projects_text = " ".join(cand.get("projects", [])).lower()
        work_exp = cand.get("work_experience", [])
        titles = [job.get("title", "") for job in work_exp]

        domain = cand.get("preferred_domain", "").strip()
        domain_req = DOMAIN_REQUIREMENTS.get(domain, {"skills": [], "job_titles": [], "project_keywords": []})

        # Feature calculations
        matched_skills = skills.intersection(set(domain_req.get("skills", [])))
        skill_match_ratio = len(matched_skills) / max(1, len(domain_req.get("skills", [])))

        relevant_experience = sum(
            job.get("years", 0) for job in work_exp if job.get("title") in domain_req.get("job_titles", [])
        )

        project_match = int(any(kw.lower() in projects_text for kw in domain_req.get("project_keywords", [])))

        record = {
            "test_score": cand.get("test_score", 0),
            "skill_match_ratio": skill_match_ratio,
            "relevant_experience": relevant_experience,
            "project_match": project_match,
            "preferred_domain": domain.lower() if domain else "unknown",
            "skills_text": " ".join(skills).lower(),
            "projects_text": projects_text,
            "titles_text": " ".join(titles).lower()
        }
        records.append(record)

    return pd.DataFrame(records)


def predict_from_json(json_data, model_path="catboost_resume_model_updated.cbm"):
    """Predict labels for candidates from JSON data."""
    feature_cols = [
        "test_score", "skill_match_ratio", "relevant_experience", "project_match",
        "preferred_domain", "skills_text", "projects_text", "titles_text"
    ]

    df = json_to_df(json_data)

    # Clean features
    text_cols = ["skills_text", "projects_text", "titles_text"]
    cat_cols  = ["preferred_domain"]
    num_cols  = ["test_score", "skill_match_ratio", "relevant_experience", "project_match"]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in num_cols:
        df[col] = df[col].fillna(0.0)

    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Use Pool for prediction to handle categorical/text features properly
    cat_idx  = [feature_cols.index(c) for c in cat_cols]
    text_idx = [feature_cols.index(c) for c in text_cols]
    test_pool = Pool(df[feature_cols], cat_features=cat_idx, text_features=text_idx)

    # Predict
    predictions = model.predict(test_pool).flatten()
    return predictions


if __name__ == "__main__":
    json_file_path = "candidates_prediction.json"

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = predict_from_json(data)
    for i, p in enumerate(preds):
        print(f"Candidate {i+1}: Predicted label = {p}")
