"""
Data downloader and verified benchmark aggregator for Retrospect Laya Evaluation.
Fetches verified ground truth oracles for Java, Python, JavaScript, and TypeScript.
"""
import os
import json
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already downloaded: {dest_path.name}")
        return
    print(f"Downloading {url} -> {dest_path.name}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"Saved: {dest_path.name} ({dest_path.stat().st_size} bytes)")

def fetch_datasets():
    # 1. CodeTracker Java Method Tracking Test & Training CSVs
    codetracker_base = "https://raw.githubusercontent.com/jodavimehran/code-tracker/master/experiments/tracking-accuracy/method/tracker"
    download_file(f"{codetracker_base}/detailed-tracker-test.csv", RAW_DIR / "codetracker_test.csv")
    download_file(f"{codetracker_base}/detailed-tracker-training.csv", RAW_DIR / "codetracker_train.csv")

    # 2. RefDiff JS Evaluation Data
    refdiff_js_url = "https://raw.githubusercontent.com/aserg-ufmg/RefDiff/master/refdiff-evaluation/data/js-evaluation/data.csv"
    download_file(refdiff_js_url, RAW_DIR / "refdiff_js.tsv")

    # 3. RefactoringMiner Multi-language Oracles
    rminer_base = "https://raw.githubusercontent.com/tsantalis/RefactoringMiner/master/src/test/resources/oracle"
    download_file(f"{rminer_base}/python-dataset/data.json", RAW_DIR / "rminer_python.json")
    download_file(f"{rminer_base}/javascript-dataset/data.json", RAW_DIR / "rminer_javascript.json")

    # 4. PyRef Python Dataset
    pyref_url = "https://raw.githubusercontent.com/PyRef/PyRef/main/data/dataset.csv"
    download_file(pyref_url, RAW_DIR / "pyref_python.csv")

    print("\nDataset downloads complete!")

if __name__ == "__main__":
    fetch_datasets()
