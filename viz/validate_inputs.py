"""Validate all delivered features and compare their digests with P5's received files."""
import json

from eval.p5_common import REPO_ROOT, discover_features, inspect_features, write_json, sha256


def validate():
    root = REPO_ROOT/"features"
    result = inspect_features(root, discover_features(root), REPO_ROOT/"splits")
    receipt = REPO_ROOT/"results/p5_feature_check.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8-sig"))
        expected = {r["path"]: r["sha256"] for r in previous["files"]}
        mismatches = [r["path"] for r in result["files"] if expected.get(r["path"]) != r["sha256"]]
        result["p5_receipt"] = {"path": "results/p5_feature_check.json", "sha256": sha256(receipt),
                                "matched": not mismatches, "mismatches": mismatches}
    else:
        result["p5_receipt"] = None
    result["passed"] = result["complete"] and (result["p5_receipt"] is None or result["p5_receipt"]["matched"])
    write_json(REPO_ROOT/"results/p9/input_validation.json", result)
    if not result["passed"]:
        raise ValueError("Incomplete feature matrix or mismatched upstream hashes; see results/p9/input_validation.json")
    return result


if __name__ == "__main__":
    result = validate()
    print(f"Validated {result['available_files']} files; official splits checked; P5 hashes matched: {None if result['p5_receipt'] is None else result['p5_receipt']['matched']}")
