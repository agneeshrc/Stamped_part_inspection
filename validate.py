"""Quick validation: compare pipeline results against ground truth."""
import json
import glob
import sys

results_dir = "results"
truth_dir = "data/synthetic"

pass_count = 0
fail_count = 0
errors = []

for truth_path in sorted(glob.glob(f"{truth_dir}/*.json")):
    name = truth_path.replace("\\", "/").split("/")[-1].replace(".json", "")
    result_path = f"{results_dir}/{name}.json"
    
    try:
        with open(truth_path) as f:
            truth = json.load(f)
        with open(result_path) as f:
            result = json.load(f)
    except FileNotFoundError:
        errors.append(f"{name}: RESULT FILE MISSING")
        fail_count += 1
        continue

    expected_status = truth.get("expected_status")
    expected_reason = truth.get("expected_reason")
    
    actual_status = result.get("inspection_status")
    actual_reason = result.get("reject_reason")
    
    if actual_status == expected_status and actual_reason == expected_reason:
        pass_count += 1
        print(f"  OK   {name:45s} {actual_status}")
    else:
        fail_count += 1
        errors.append(
            f"{name}: expected={expected_status}/{expected_reason}  "
            f"got={actual_status}/{actual_reason}"
        )
        print(f"  FAIL {name:45s} exp={expected_status}/{expected_reason}  got={actual_status}/{actual_reason}")

print(f"\n{'='*60}")
print(f"PASS: {pass_count}  FAIL: {fail_count}")
if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  - {e}")
sys.exit(1 if fail_count > 0 else 0)
