# import random
# import json

# # Load domain requirements from external file
# with open("domain_requirements.json", "r", encoding="utf-8") as f:
#     DOMAIN_REQUIREMENTS = json.load(f)

# domains = list(DOMAIN_REQUIREMENTS.keys())

# # -------- Generate a candidate profile (without label) --------
# def generate_candidate_profile(domain):
#     requirements = DOMAIN_REQUIREMENTS[domain]

#     # --- Work experience logic ---
#     work_experience = []
#     # 30% chance the candidate is a fresher
#     if random.random() > 0.3:
#         num_jobs = random.randint(1, 3)
#         for _ in range(num_jobs):
#             job_title = random.choice(requirements.get("job_titles", ["Professional"]))
#             years = random.randint(1, 3)
#             work_experience.append({"title": job_title, "years": years})

#     profile = {
#         "skills": random.sample(requirements["skills"], k=random.randint(1, 3)),
#         "projects": [f"{random.choice(requirements['project_keywords'])} project"],
#         "work_experience": work_experience,  # can be [] if fresher
#         "test_score": random.randint(40, 95),
#         "preferred_domain": domain
#     }
#     return profile

# # -------- Evaluate candidate profile to assign label --------
# def evaluate_candidate(candidate):
#     domain = candidate["preferred_domain"]
#     requirements = DOMAIN_REQUIREMENTS[domain]

#     skills = set(candidate.get("skills", []))
#     projects = " ".join(candidate.get("projects", [])).lower()
#     work_exp = candidate.get("work_experience", [])
#     years = sum(job.get("years", 0) for job in work_exp)
#     test_score = candidate["test_score"]

#     matched_skills = skills.intersection(requirements["skills"])
#     project_match = any(kw in projects for kw in requirements["project_keywords"])

#     # ✅ Labeling logic
#     if len(matched_skills) >= 2 and (project_match or years >= 1) and test_score >= requirements["min_score"]:
#         return "fit"
#     elif len(matched_skills) >= 2 and (project_match or years >= 1):
#         return "partial"
#     else:
#         # Check if skills match another domain better → suggest
#         for alt_domain, req in DOMAIN_REQUIREMENTS.items():
#             if alt_domain == domain:
#                 continue
#             if skills.intersection(req["skills"]) and test_score >= req["min_score"]:
#                 return "suggest"
#         return "no_fit"

# # -------- Generate dataset --------
# total_samples = 2000  # change to 5000, 10000, etc.
# dataset = []

# for _ in range(total_samples):
#     domain = random.choice(domains)
#     profile = generate_candidate_profile(domain)
#     profile["label"] = evaluate_candidate(profile)  # ✅ Auto labeling
#     dataset.append(profile)

# # -------- Save dataset --------
# output_path = "candidates_dataset.json"
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(dataset, f, indent=2)

# print(f"✅ Dataset saved to {output_path} with {len(dataset)} samples.")



import random
import json

def generate_dataset(domain_path, output_path, total_samples):
    # Load domain requirements
    with open(domain_path, "r", encoding="utf-8") as f:
        DOMAIN_REQUIREMENTS = json.load(f)

    domains = list(DOMAIN_REQUIREMENTS.keys())

    # -------- Generate a candidate profile (without label) --------
    def generate_candidate_profile(domain):
        requirements = DOMAIN_REQUIREMENTS[domain]

        # --- Work experience logic ---
        work_experience = []
        if random.random() > 0.3:  # 30% chance fresher
            num_jobs = random.randint(1, 3)
            for _ in range(num_jobs):
                job_title = random.choice(requirements.get("job_titles", ["Professional"]))
                years = random.randint(1, 3)
                work_experience.append({"title": job_title, "years": years})

        # --- Skills logic ---
        skills = random.sample(requirements["skills"], k=random.randint(1, min(3, len(requirements["skills"]))))
        other_skills = ["communication", "teamwork", "problem solving", "critical thinking"]
        # Add some irrelevant skills
        skills += random.sample(other_skills, k=random.randint(0, 2))

        # --- Projects logic ---
        num_projects = random.randint(1, 2)
        projects = []
        for _ in range(num_projects):
            # 50% chance project is relevant
            if random.random() > 0.5:
                projects.append(f"{random.choice(requirements['project_keywords'])} project")
            else:
                projects.append(f"{random.choice(other_skills)} project")

        profile = {
            "skills": skills,
            "projects": projects,
            "work_experience": work_experience,
            "test_score": random.randint(40, 95),
            "preferred_domain": domain
        }
        return profile

    # -------- Evaluate candidate profile to assign label --------
    def evaluate_candidate(candidate):
        domain = candidate["preferred_domain"]
        requirements = DOMAIN_REQUIREMENTS[domain]

        skills = set(candidate.get("skills", []))
        projects = " ".join(candidate.get("projects", [])).lower()
        work_exp = candidate.get("work_experience", [])
        years = sum(job.get("years", 0) for job in work_exp)
        test_score = candidate["test_score"]

        # --- Feature evaluation ---
        matched_skills = skills.intersection(requirements["skills"])
        skill_match_ratio = len(matched_skills) / len(requirements["skills"])  # percentage of relevant skills
        project_match = any(kw in projects for kw in requirements["project_keywords"])
        experience_relevant = sum(job.get("years", 0) for job in work_exp 
                                  if job.get("title") in requirements.get("job_titles", []))

        # ✅ Labeling logic
        if skill_match_ratio >= 0.5 and (project_match or experience_relevant >= 1) and test_score >= requirements["min_score"]:
            return "fit"
        elif skill_match_ratio >= 0.5 and (project_match or experience_relevant >= 1):
            return "partial"
        else:
            # Check if candidate fits another domain better → suggest
            for alt_domain, req in DOMAIN_REQUIREMENTS.items():
                if alt_domain == domain:
                    continue
                if skills.intersection(req["skills"]) and test_score >= req["min_score"]:
                    return "suggest"
            return "no_fit"

    # -------- Generate dataset --------
    dataset = []
    for _ in range(total_samples):
        domain = random.choice(domains)
        profile = generate_candidate_profile(domain)
        profile["label"] = evaluate_candidate(profile)
        dataset.append(profile)

    # -------- Save dataset --------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Dataset saved to {output_path} with {len(dataset)} samples.")




