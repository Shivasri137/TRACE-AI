"""Evaluate manually labeled predictions in evaluation_results.json."""
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
path = Path("evaluation_results.json")
if not path.exists():
    raise SystemExit("Create evaluation_results.json with manually labeled predictions first.")
records = json.loads(path.read_text(encoding="utf-8"))
if not records:
    raise SystemExit("The evaluation dataset is empty.")
y_true = [str(row["expected"]).upper() for row in records]
y_pred = [str(row["predicted"]).upper() for row in records]
invalid = sorted((set(y_true) | set(y_pred)) - set(LABELS))
if invalid:
    raise SystemExit(f"Unsupported labels: {invalid}. Valid labels: {LABELS}")
print(f"Examples: {len(records)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=LABELS))
