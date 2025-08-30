# import random
# import json
# import time
# from collections import defaultdict

# def generate_dataset(domain_path, output_path, total_samples):
#     start_time = time.time()
#     print("⏳ Starting dataset generation...")

#     # Load domain requirements
#     with open(domain_path, "r", encoding="utf-8") as f:
#         DOMAIN_REQUIREMENTS = json.load(f)
#     domains = list(DOMAIN_REQUIREMENTS.keys())

#     # Track label counts
#     label_counts = defaultdict(int)
#     domain_label_counts = {d: defaultdict(int) for d in domains}

#     # --- Generate a candidate profile ---
#     def generate_candidate_profile(domain):
#         req = DOMAIN_REQUIREMENTS[domain]

#         # Work experience
#         work_exp = []
#         if random.random() > 0.3:  # 30% chance to be fresher
#             for _ in range(random.randint(1, 3)):
#                 work_exp.append({
#                     "title": random.choice(req.get("job_titles", ["Professional"])),
#                     "years": random.randint(1, 3)
#                 })

#         # In generate_candidate_profile
#         skills = random.sample(req["skills"], k=random.randint(max(2, len(req["skills"])//2), len(req["skills"])))

#         other_skills = ["communication", "teamwork", "problem solving", "critical thinking"]
#         skills += random.sample(other_skills, k=random.randint(0, 2))

#         projects = []
#         for _ in range(random.randint(1, 2)):
#             if random.random() > 0.3:  # 70% chance to match domain
#                 projects.append(f"{random.choice(req['project_keywords'])} project")
#             else:
#                 projects.append(f"{random.choice(other_skills)} project")

#         return {
#             "skills": skills,
#             "projects": projects,
#             "work_experience": work_exp,
#             "test_score": random.randint(req["min_score"] - 5, 95),
#             "preferred_domain": domain
#         }

#     # --- Evaluate candidate profile ---
#     def evaluate_candidate(candidate):
#         domain = candidate["preferred_domain"]
#         req = DOMAIN_REQUIREMENTS[domain]

#         skills = set(candidate["skills"])
#         projects_text = " ".join(candidate["projects"]).lower()
#         work_exp = candidate["work_experience"]
#         total_years = sum(job["years"] for job in work_exp)
#         relevant_exp = sum(job["years"] for job in work_exp if job["title"] in req.get("job_titles", []))
#         test_score = candidate["test_score"]

#         matched_skills = skills.intersection(req["skills"])
#         skill_ratio = len(matched_skills) / len(req["skills"])
#         project_match = any(kw in projects_text for kw in req["project_keywords"])

#         # Label rules
#         if skill_ratio >= 0.5 and (project_match or relevant_exp >= 1) and test_score >= req["min_score"]:
#             return "fit"
#         elif skill_ratio >= 0.4 and (project_match or relevant_exp >= 0.5) and test_score >= (req["min_score"] - 10 ):
#             return "partial"
#         else:
#             # Suggest for other domains
#             for alt_domain, alt_req in DOMAIN_REQUIREMENTS.items():
#                 if alt_domain == domain:
#                     continue
#                 if skills.intersection(alt_req["skills"]) and test_score >= alt_req["min_score"]:
#                     return "suggest"
#             return "no_fit"

#     # --- Generate dataset ---
#     dataset = []
#     for i in range(total_samples):
#         domain = random.choice(domains)
#         candidate = generate_candidate_profile(domain)
#         candidate["label"] = evaluate_candidate(candidate)
#         dataset.append(candidate)

#         label_counts[candidate["label"]] += 1
#         domain_label_counts[domain][candidate["label"]] += 1

#         # Minimal progress log
#         if (i + 1) % max(1, total_samples // 10) == 0:
#             print(f"🔹 {i + 1}/{total_samples} samples generated")

#     # Save dataset
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(dataset, f, indent=2)

#     # Summary
#     print("\n📊 Label counts:", dict(label_counts))
#     print("📊 Domain-wise label counts:")
#     for domain, counts in domain_label_counts.items():
#         print(f"{domain}: {dict(counts)}")

#     print(f"\n✅ Dataset generation finished in {time.time() - start_time:.2f} seconds.")
#     return dataset, label_counts, domain_label_counts


import random
import json
import time
from collections import defaultdict

def generate_dataset(domain_path, output_path, total_samples, seed=42):
    random.seed(seed)
    start_time = time.time()
    print("⏳ Starting dataset generation...")

    # Load domain requirements
    with open(domain_path, "r", encoding="utf-8") as f:
        DOMAIN_REQUIREMENTS = json.load(f)
    domains = list(DOMAIN_REQUIREMENTS.keys())

    # Track label counts
    label_counts = defaultdict(int)
    domain_label_counts = {d: defaultdict(int) for d in domains}

    # --- Generate a candidate profile ---
    def generate_candidate_profile(domain):
        req = DOMAIN_REQUIREMENTS[domain]

        # Work experience
        work_exp = []
        if random.random() > 0.3:  # 30% chance to be fresher
            for _ in range(random.randint(1, 3)):
                work_exp.append({
                    "title": random.choice(req.get("job_titles", ["Professional"])),
                    "years": random.randint(1, 3)
                })

        # Skills
        skills = random.sample(req["skills"], k=random.randint(max(2, len(req["skills"])//2), len(req["skills"])))
        other_skills = ["communication", "teamwork", "problem solving", "critical thinking"]
        skills += random.sample(other_skills, k=random.randint(0, 2))

        # Projects
        projects = []
        for _ in range(random.randint(1, 2)):
            if random.random() > 0.3:
                projects.append(f"{random.choice(req['project_keywords'])} project")
            else:
                projects.append(f"{random.choice(other_skills)} project")

        # Test score
        test_score = random.randint(req["min_score"], 95)

        return {
            "skills": skills,
            "projects": projects,
            "work_experience": work_exp,
            "test_score": test_score,
            "preferred_domain": domain
        }

    # --- Evaluate candidate profile ---
    def evaluate_candidate(candidate):
        domain = candidate["preferred_domain"]
        req = DOMAIN_REQUIREMENTS[domain]

        skills = set(candidate["skills"])
        projects_text = " ".join(candidate["projects"]).lower()
        work_exp = candidate["work_experience"]
        total_years = sum(job["years"] for job in work_exp)
        relevant_exp = sum(job["years"] for job in work_exp if job["title"] in req.get("job_titles", []))
        test_score = candidate["test_score"]

        matched_skills = skills.intersection(req["skills"])
        skill_ratio = len(matched_skills) / max(1, len(req["skills"]))
        project_match = any(kw in projects_text for kw in req["project_keywords"])

        # --- Threshold-based label assignment ---
        if test_score >= 80 and skill_ratio > 0.35 and relevant_exp > 1:
            return "fit"
        elif test_score >= 70 and skill_ratio > 0.25 and relevant_exp >= 1:
            return "partial"
        else:
            # Check if candidate fits another domain
            for alt_domain, alt_req in DOMAIN_REQUIREMENTS.items():
                if alt_domain == domain:
                    continue
                if skills.intersection(alt_req["skills"]) and test_score >= alt_req["min_score"]:
                    return "suggest"
            return "no_fit"

    # --- Generate dataset ---
    dataset = []
    for i in range(total_samples):
        domain = random.choice(domains)
        candidate = generate_candidate_profile(domain)
        candidate["label"] = evaluate_candidate(candidate)
        dataset.append(candidate)

        label_counts[candidate["label"]] += 1
        domain_label_counts[domain][candidate["label"]] += 1

        # Progress log
        if (i + 1) % max(1, total_samples // 10) == 0:
            print(f"🔹 {i + 1}/{total_samples} samples generated")

    # Save dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    print("\n📊 Label counts:", dict(label_counts))
    print("📊 Domain-wise label counts:")
    for domain, counts in domain_label_counts.items():
        print(f"{domain}: {dict(counts)}")

    print(f"\n✅ Dataset generation finished in {time.time() - start_time:.2f} seconds.")
    return dataset, label_counts, domain_label_counts
