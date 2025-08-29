# import pandas as pd
# import json

# # Load your dataset JSON
# with open("candidates_dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Helper: flatten JSON fields into usable features
# def json_to_row(sample):
#     skills = sample.get("skills", [])
#     projects = sample.get("projects", [])
#     work_exp = sample.get("work_experience", [])

#     total_years = sum(job.get("years", 0) for job in work_exp)
#     titles = [job.get("title", "") for job in work_exp]

#     return {
#         "test_score": sample.get("test_score", None),
#         "total_years": total_years,
#         "n_skills": len(skills),
#         "n_projects": len(projects),
#         "preferred_domain": sample.get("preferred_domain", "").lower(),
#         "skills_text": " ".join(skills).lower(),
#         "projects_text": " ".join(projects).lower(),
#         "titles_text": " ".join(titles).lower(),
#         "label": sample.get("label", "").lower()
#     }

# # Build dataframe
# rows = [json_to_row(x) for x in data]
# df = pd.DataFrame(rows)

# # Save both CSV and Parquet
# output_csv = "candidates_dataset.csv"
# output_parquet = "candidates_dataset.parquet"

# df.to_csv(output_csv, index=False)
# df.to_parquet(output_parquet, index=False)

# print(f"✅ Dataset saved as {output_csv} and {output_parquet}")



import pandas as pd
import json

def update_dataset(dataset_path, domain_path, output_csv, output_parquet):
    # Load your dataset JSON
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load domain requirements for feature relevance
    with open(domain_path, "r", encoding="utf-8") as f:
        DOMAIN_REQUIREMENTS = json.load(f)

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
    rows = [json_to_row(x) for x in data]
    df = pd.DataFrame(rows)

    # Save both CSV and Parquet
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"✅ Updated dataset saved as {output_csv} and {output_parquet}")
