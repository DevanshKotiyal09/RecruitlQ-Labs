"""
Full integration test — run from project root.
Tests the complete ranking pipeline end-to-end against sample data.
"""
import sys
import json
import os

sys.path.insert(0, ".")

from ranker.jd_parser import load_jd_requirements
from ranker.feature_extractor import extract_features
from ranker.anti_trap import evaluate_trust, detect_behavioral_twins
from ranker.scorer import compute_composite_score
from ranker.behavioral_analyser import compute_behavioral_composite
from ranker.ranker import ScoredCandidate, select_top_n, assign_ranks
from ranker.reason_generator import generate_reasons_for_top100
from ranker.output_assembler import write_submission_csv, validate_csv_locally

jd = load_jd_requirements()
print(f"[1] JD loaded: {len(jd.must_have_skills)} must-have skills, "
      f"{len(jd.preferred_skills)} preferred skills, "
      f"{len(jd.consulting_firms)} consulting firms")

sample_path = (
    "Data Set/[PUB] India_runs_data_and_ai_challenge/"
    "India_runs_data_and_ai_challenge/sample_candidates.json"
)
with open(sample_path, encoding="utf-8") as f:
    candidates = json.load(f)
print(f"[2] Loaded {len(candidates)} sample candidates")

feat_idx, trust_idx, cand_idx = {}, {}, {}
all_scored = []
twin_pairs = []

for cand in candidates:
    cid = cand["candidate_id"]
    cand_idx[cid] = cand
    feat = extract_features(cand, jd)
    trust = evaluate_trust(cand, feat)
    feat_idx[cid] = feat
    trust_idx[cid] = trust
    twin_pairs.append(({"redrob_signals": cand.get("redrob_signals", {})}, feat))
    composite, conf = compute_composite_score(feat, trust, jd)
    beh = compute_behavioral_composite(feat)
    all_scored.append(ScoredCandidate(
        candidate_id=cid,
        composite_score=composite,
        confidence_score=conf,
        behavioral_composite=beh,
    ))

print(f"[3] Feature extraction and scoring complete for {len(all_scored)} candidates")

# Verify new features are populated
sample_feat = feat_idx[list(feat_idx.keys())[0]]
assert hasattr(sample_feat, "preferred_skill_count"), "Missing preferred_skill_count"
assert hasattr(sample_feat, "preferred_skill_match_ratio"), "Missing preferred_skill_match_ratio"
assert hasattr(sample_feat, "preferred_skill_corroborated"), "Missing preferred_skill_corroborated"
assert hasattr(sample_feat, "python_quality_signal"), "Missing python_quality_signal"
print(f"[3a] New feature fields present: OK")
print(f"     Sample: preferred_skill_count={sample_feat.preferred_skill_count}, "
      f"preferred_skill_corroborated={sample_feat.preferred_skill_corroborated}, "
      f"python_quality_signal={sample_feat.python_quality_signal:.3f}")

twin_flags = detect_behavioral_twins(twin_pairs)
print(f"[4] Twin detections: {len(twin_flags)}")

# Select top N (limited to pool size in sample)
n = len(all_scored)
top = select_top_n(all_scored, n=n)
ranked = assign_ranks(top)

print("[5] Top 10 ranked candidates:")
for rank, s in ranked[:10]:
    feat = feat_idx[s.candidate_id]
    profile = cand_idx[s.candidate_id]["profile"]
    title = profile["current_title"][:30]
    print(f"    #{rank:3d}: {s.candidate_id} | {title:<30} | "
          f"score={s.composite_score:.4f} | "
          f"retr={feat.retrieval_career_evidence:.3f} | "
          f"pref_corr={feat.preferred_skill_corroborated} | "
          f"py_qual={feat.python_quality_signal:.3f} | "
          f"cons_only={feat.consulting_only_flag}")

# Validate top candidate has some AI signal
top1_feat = feat_idx[ranked[0][1].candidate_id]
assert (
    top1_feat.current_title_ai_relevance > 0 or
    top1_feat.retrieval_career_evidence > 0 or
    top1_feat.jd_core_skill_corroborated > 0 or
    top1_feat.career_direction_score > 0
), "Top candidate should have at least some AI relevance signal"
print("[5a] Top candidate has AI relevance: OK")

# Verify score monotonicity
for i in range(len(ranked) - 1):
    r1, s1 = ranked[i]
    r2, s2 = ranked[i + 1]
    assert s1.composite_score >= s2.composite_score, (
        f"Score not monotonic at ranks {r1},{r2}: {s1.composite_score} < {s2.composite_score}"
    )
print("[5b] Score monotonicity: OK")

# Generate reasons
reasons = generate_reasons_for_top100(ranked, cand_idx, feat_idx, trust_idx, jd)
print("[6] Sample reasons:")
for rank, s in ranked[:5]:
    r = reasons.get(s.candidate_id, "MISSING")
    assert len(r) > 5, f"Reason too short for rank {rank}"
    print(f"    [{rank}] {r[:120]}")

# Write CSV
errors = write_submission_csv(ranked, reasons, "test_submission_output.csv", valid_ids=None)
if errors:
    # Only fail on non-row-count errors (sample has < 100 candidates)
    real_errors = [e for e in errors if "100 rows" not in e]
    if real_errors:
        print("CSV WRITE ERRORS:", real_errors)
        sys.exit(1)
print(f"[7] CSV written (errors={errors})")

val_errors = validate_csv_locally("test_submission_output.csv")
# With < 100 candidates the row-count check will fail — that's expected
structural_errors = [e for e in val_errors if "100" not in e]
if structural_errors:
    print("CSV STRUCTURAL ERRORS:", structural_errors)
    sys.exit(1)
print(f"[7a] CSV structural validation: OK (total issues={len(val_errors)}, "
      f"all row-count related for sample size)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

# Cleanup
if os.path.exists("test_submission_output.csv"):
    os.remove("test_submission_output.csv")
