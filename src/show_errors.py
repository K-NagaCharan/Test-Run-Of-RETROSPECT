"""
Print error cases helper script
"""
import json

def show_errors():
    with open("results/tables/error_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    laya_cases = [c for c in cases if "Laya" in c["model"]]
    print(f"Total Laya error cases logged: {len(laya_cases)}")
    for c in laya_cases[:10]:
        print("-" * 50)
        print(f"Model: {c['model']} | Lang: {c['language']} | Change: {c['change_type']} | Err: {c['error_type']}")
        print(f"Target: {c['target_name']} | GT: {c['ground_truth']} | Pred: {c['prediction']} | Conf: {c['confidence']:.4f}")
        print("Snippet:", c['target_snippet'].replace('\n', ' ')[:120])

if __name__ == "__main__":
    show_errors()
