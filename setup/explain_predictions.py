# import json
# import pandas as pd

# # ---------------- Load Files ----------------
# with open("domain_requirements.json", "r") as f:
#     domain_rules = json.load(f)

# with open("candidates_prediction.json", "r") as f:
#     candidates = json.load(f)

# df = pd.read_csv(r"C:\Users\pavan\Desktop\ai\setup\output\final_output.csv")  
# predictions = df["predicted_label"].tolist()


# # ---------------- Helpers ----------------
# def fit_intensity_from_label(prediction):
#     mapping = {
#         "fit": "strong fit",
#         "suggest": "moderate fit",
#         "partial": "partial fit",
#         "no_fit": "low fit"
#     }
#     return mapping.get(prediction, "partial fit")


# def custom_reason(candidate, strengths, weaknesses, prediction, missing_skills_full):
#     """Build a clear reason aligned with predicted_label and actual missing skills"""
#     candidate.setdefault("projects", [])
#     candidate.setdefault("skills", [])
#     candidate.setdefault("work_experience", [])
#     candidate.setdefault("test_score", 0)
#     candidate.setdefault("preferred_domain", "Unknown Domain")

#     skills_str = ", ".join(candidate.get("skills", [])) or "N/A"
#     projects_str = ", ".join(candidate.get("projects", [])) or "N/A"
#     experience_str = ", ".join([exp.get("title", "N/A") for exp in candidate.get("work_experience", [])]) \
#                      if candidate.get("work_experience") else "N/A"

#     sentences = []

#     # ---------------- Strengths ----------------
#     if any("test score" in s.lower() for s in strengths):
#         sentences.append(f"You demonstrated knowledge through your test score of {candidate['test_score']}.")
#     if any("skills matched" in s.lower() for s in strengths):
#         sentences.append(f"You have relevant skills ({skills_str}) that align well with this domain.")
#     if any("relevant experience" in s.lower() for s in strengths):
#         sentences.append(f"Your experience as {experience_str} adds practical value.")
#     if any("relevant projects" in s.lower() for s in strengths):
#         sentences.append(f"You worked on projects like {projects_str}, which are directly relevant.")

#     # ---------------- Weaknesses ----------------
#     if missing_skills_full:
#         sentences.append(f"Your skills are not fully aligned. You require {', '.join(missing_skills_full)} to improve your suitability.")
#     if any("no relevant domain experience" in w.lower() for w in weaknesses):
#         sentences.append("You have limited or no professional experience, which is important for practical application.")
#     if any("test score" in w.lower() for w in weaknesses):
#         sentences.append(f"Your test score ({candidate['test_score']}) shows some knowledge, but it is below the expected level.")
#     if any("no relevant projects" in w.lower() for w in weaknesses):
#         sentences.append("No relevant projects were found. Building domain-specific projects will help.")

#     intensity = fit_intensity_from_label(prediction)
#     return f"You are evaluated as a {intensity} for {candidate['preferred_domain']}. " + " ".join(sentences)


# # ---------------- Explanation Logic ----------------
# def explain_prediction(candidate, domain_rules, prediction):
#     domain_name = candidate.get("preferred_domain", "")
#     if domain_name not in domain_rules:
#         return {
#             "preferred_domain": domain_name,
#             "predicted_label": prediction,
#             "strengths": [],
#             "weaknesses": ["Domain rules not found."],
#             "reason": "No evaluation possible for this domain."
#         }

#     domain = domain_rules[domain_name]
#     strengths, weaknesses = [], []

#     # ✅ Test score
#     test_score = candidate.get("test_score", 0)
#     min_score = domain.get("min_score", 0)
#     if test_score >= min_score:
#         strengths.append(f"Test score {test_score} meets/exceeds minimum {min_score}")
#     else:
#         weaknesses.append(f"Test score too low ({test_score} < {min_score})")

#     # ✅ Skills
#     candidate_skills = set(s.lower() for s in candidate.get("skills", []))
#     required_skills = set(s.lower() for s in domain.get("skills", []))
#     matched_skills = candidate_skills & required_skills
#     missing_skills = required_skills - candidate_skills
#     missing_skills_full = list(missing_skills)
#     if matched_skills:
#         strengths.append("Skills matched: " + ", ".join(matched_skills))
#     if missing_skills:
#         weaknesses.append("Skills missing: " + ", ".join(list(missing_skills)[:5]) + ("..." if len(missing_skills) > 5 else ""))

#     # ✅ Projects
#     projects = candidate.get("projects", [])
#     project_keywords = [kw.lower() for kw in domain.get("project_keywords", [])]
#     relevant_projects = [p for p in projects if any(kw in p.lower() for kw in project_keywords)]
#     if relevant_projects:
#         strengths.append("Relevant projects: " + ", ".join(relevant_projects[:3]))
#     else:
#         weaknesses.append("No relevant projects found")

#     # ✅ Experience
#     exp_titles = [exp.get("title", "").lower() for exp in candidate.get("work_experience", [])]
#     domain_titles = [t.lower() for t in domain.get("job_titles", [])]
#     matched_exp = [exp.get("title") for exp in candidate.get("work_experience", []) if any(dt in exp.get("title", "").lower() for dt in domain_titles)]
#     if matched_exp:
#         strengths.append("Relevant experience: " + ", ".join(matched_exp))
#     else:
#         weaknesses.append("No relevant domain experience")

#     # ✅ Return JSON
#     return {
#         "preferred_domain": domain_name,
#         "predicted_label": prediction,
#         "strengths": strengths,
#         "weaknesses": weaknesses,
#         "reason": custom_reason(candidate, strengths, weaknesses, prediction, missing_skills_full)
#     }


# # ---------------- Generate Explanations ----------------
# results = []
# for idx, cand in enumerate(candidates):
#     pred_label = predictions[idx] if idx < len(predictions) else "no_fit"
#     explanation = explain_prediction(cand, domain_rules, pred_label)
#     results.append(explanation)
    
#     # Print each candidate's explanation in JSON format
#     print(json.dumps(explanation, indent=4))

# # ---------------- Save JSON ----------------
# with open("explanations.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("✅ Explanations saved to explanations.json")


import json
import pandas as pd

def generate_explanations(candidates_json, domain_json, predictions_csv, output_json="explanations.json"):
    """
    Generate candidate explanation JSON based on predictions and domain rules.

    Args:
        candidates_json (str or Path): Path to candidates JSON file.
        domain_json (str or Path): Path to domain requirements JSON file.
        predictions_csv (str or Path): CSV file with predicted labels.
        output_json (str or Path): Output path to save explanations.

    Returns:
        list: List of explanation dicts for each candidate.
    """

    # ---------------- Load Files ----------------
    with open(domain_json, "r") as f:
        domain_rules = json.load(f)

    with open(candidates_json, "r") as f:
        candidates = json.load(f)

    df = pd.read_csv(predictions_csv)
    predictions = df["predicted_label"].tolist()

    # ---------------- Helpers ----------------
    def fit_intensity_from_label(prediction):
        mapping = {
            "fit": "strong fit",
            "suggest": "moderate fit",
            "partial": "partial fit",
            "no_fit": "low fit"
        }
        return mapping.get(prediction, "partial fit")

    def custom_reason(candidate, strengths, weaknesses, prediction, missing_skills_full):
        candidate.setdefault("projects", [])
        candidate.setdefault("skills", [])
        candidate.setdefault("work_experience", [])
        candidate.setdefault("test_score", 0)
        candidate.setdefault("preferred_domain", "Unknown Domain")

        skills_str = ", ".join(candidate.get("skills", [])) or "N/A"
        projects_str = ", ".join(candidate.get("projects", [])) or "N/A"
        experience_str = ", ".join([exp.get("title", "N/A") for exp in candidate.get("work_experience", [])]) \
                         if candidate.get("work_experience") else "N/A"

        sentences = []

        # Strengths
        if any("test score" in s.lower() for s in strengths):
            sentences.append(f"You demonstrated knowledge through your test score of {candidate['test_score']}.")
        if any("skills matched" in s.lower() for s in strengths):
            sentences.append(f"You have relevant skills ({skills_str}) that align well with this domain.")
        if any("relevant experience" in s.lower() for s in strengths):
            sentences.append(f"Your experience as {experience_str} adds practical value.")
        if any("relevant projects" in s.lower() for s in strengths):
            sentences.append(f"You worked on projects like {projects_str}, which are directly relevant.")

        # Weaknesses
        if missing_skills_full:
            sentences.append(f"Your skills are not fully aligned. You require {', '.join(missing_skills_full)} to improve your suitability.")
        if any("no relevant domain experience" in w.lower() for w in weaknesses):
            sentences.append("You have limited or no professional experience, which is important for practical application.")
        if any("test score" in w.lower() for w in weaknesses):
            sentences.append(f"Your test score ({candidate['test_score']}) shows some knowledge, but it is below the expected level.")
        if any("no relevant projects" in w.lower() for w in weaknesses):
            sentences.append("No relevant projects were found. Building domain-specific projects will help.")

        intensity = fit_intensity_from_label(prediction)
        return f"You are evaluated as a {intensity} for {candidate['preferred_domain']}. " + " ".join(sentences)

    # ---------------- Explanation Logic ----------------
    def explain_prediction(candidate, domain_rules, prediction):
        domain_name = candidate.get("preferred_domain", "")
        if domain_name not in domain_rules:
            return {
                "preferred_domain": domain_name,
                "predicted_label": prediction,
                "strengths": [],
                "weaknesses": ["Domain rules not found."],
                "reason": "No evaluation possible for this domain."
            }

        domain = domain_rules[domain_name]
        strengths, weaknesses = [], []

        # Test score
        test_score = candidate.get("test_score", 0)
        min_score = domain.get("min_score", 0)
        if test_score >= min_score:
            strengths.append(f"Test score {test_score} meets/exceeds minimum {min_score}")
        else:
            weaknesses.append(f"Test score too low ({test_score} < {min_score})")

        # Skills
        candidate_skills = set(s.lower() for s in candidate.get("skills", []))
        required_skills = set(s.lower() for s in domain.get("skills", []))
        matched_skills = candidate_skills & required_skills
        missing_skills = required_skills - candidate_skills
        missing_skills_full = list(missing_skills)
        if matched_skills:
            strengths.append("Skills matched: " + ", ".join(matched_skills))
        if missing_skills:
            weaknesses.append("Skills missing: " + ", ".join(list(missing_skills)[:5]) + ("..." if len(missing_skills) > 5 else ""))

        # Projects
        projects = candidate.get("projects", [])
        project_keywords = [kw.lower() for kw in domain.get("project_keywords", [])]
        relevant_projects = [p for p in projects if any(kw in p.lower() for kw in project_keywords)]
        if relevant_projects:
            strengths.append("Relevant projects: " + ", ".join(relevant_projects[:3]))
        else:
            weaknesses.append("No relevant projects found")

        # Experience
        exp_titles = [exp.get("title", "").lower() for exp in candidate.get("work_experience", [])]
        domain_titles = [t.lower() for t in domain.get("job_titles", [])]
        matched_exp = [exp.get("title") for exp in candidate.get("work_experience", []) if any(dt in exp.get("title", "").lower() for dt in domain_titles)]
        if matched_exp:
            strengths.append("Relevant experience: " + ", ".join(matched_exp))
        else:
            weaknesses.append("No relevant domain experience")

        return {
            "preferred_domain": domain_name,
            "predicted_label": prediction,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "reason": custom_reason(candidate, strengths, weaknesses, prediction, missing_skills_full)
        }

    # ---------------- Generate Explanations ----------------
    results = []
    for idx, cand in enumerate(candidates):
        pred_label = predictions[idx] if idx < len(predictions) else "no_fit"
        explanation = explain_prediction(cand, domain_rules, pred_label)
        results.append(explanation)
        print(json.dumps(explanation, indent=4))

    # ---------------- Save JSON ----------------
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Explanations saved to {output_json}")
    return results
