import pandas as pd
import json
import time  # For timing measurements

def update_dataset(dataset_path, domain_path, output_csv, output_parquet):
    start_time = time.time()
    print("⏳ Starting dataset update...")

    # Load your dataset JSON
    t0 = time.time()
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    t1 = time.time()
    print(f"📂 Loaded dataset JSON ({len(data)} samples) in {t1 - t0:.4f} seconds.")

    # Load domain requirements for feature relevance
    t2 = time.time()
    with open(domain_path, "r", encoding="utf-8") as f:
        DOMAIN_REQUIREMENTS = json.load(f)
    t3 = time.time()
    print(f"📂 Loaded domain requirements in {t3 - t2:.4f} seconds.")

    # Helper: convert JSON sample to ML-ready features
    def json_to_row(sample):
        skills = set(sample.get("skills", []))
        projects = " ".join(sample.get("projects", [])).lower()
        work_exp = sample.get("work_experience", [])
        total_years = sum(job.get("years", 0) for job in work_exp)
        titles = [job.get("title", "") for job in work_exp]

        domain = sample.get("preferred_domain", "")
        domain_req = DOMAIN_REQUIREMENTS.get(domain, {})

        # --- New relevance features ---
        matched_skills = skills.intersection(domain_req.get("skills", []))
        skill_match_ratio = len(matched_skills) / max(1, len(domain_req.get("skills", [])))

        relevant_experience = sum(
            job.get("years", 0) for job in work_exp if job.get("title") in domain_req.get("job_titles", [])
        )

        project_match = int(any(kw in projects for kw in domain_req.get("project_keywords", [])))

        return {
            "test_score": sample.get("test_score", 0),
            "skill_match_ratio": skill_match_ratio,      # replaces n_skills
            "relevant_experience": relevant_experience,  # replaces total_years
            "project_match": project_match,              # replaces n_projects
            "preferred_domain": domain.lower(),
            "skills_text": " ".join(skills).lower(),
            "projects_text": projects,
            "titles_text": " ".join(titles).lower(),
            "label": sample.get("label", "").lower()
        }

    # Build dataframe
    t_build_start = time.time()
    rows = []
    for i, x in enumerate(data):
        rows.append(json_to_row(x))
        if (i + 1) % max(1, len(data)//10) == 0:
            print(f"🔹 Processed {i + 1}/{len(data)} samples...")
    df = pd.DataFrame(rows)
    t_build_end = time.time()
    print(f"⏱ Dataframe built in {t_build_end - t_build_start:.2f} seconds.")

    # Save both CSV and Parquet
    t_save_start = time.time()
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)
    t_save_end = time.time()
    print(f"💾 Dataset saved as CSV and Parquet in {t_save_end - t_save_start:.4f} seconds.")

    end_time = time.time()
    print(f"✅ Total process finished in {end_time - start_time:.2f} seconds.")
