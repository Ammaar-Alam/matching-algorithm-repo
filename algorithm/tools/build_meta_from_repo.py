import json, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_json", required=True, help="Path to repo data/questions.json")
    ap.add_argument("--export_csv", required=True, help="Path to export.csv to detect IMC columns")
    ap.add_argument("--out", default="meta.json")
    args = ap.parse_args()

    with open(args.questions_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Build meta from repo items
    meta = {}
    for q in items:
        qid = q["id"]
        dom = q.get("domain", "")
        # normalize to algorithm domain names
        if dom == "Friendship Qualities":
            dom = "Friendship"
        meta[qid] = {
            "ResponseType": q.get("responseType", "Likert5"),
            "Domain": dom,
        }

    # If export has IMC1 but meta doesn't, add it (ignored later via 0 weight)
    df = pd.read_csv(args.export_csv, nrows=1)
    for col in df.columns:
        if col.startswith("SELF_IMC1") and "IMC1" not in meta:
            meta["IMC1"] = {"ResponseType":"Likert5", "Domain":"Quality"}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.out} with {len(meta)} items.")

if __name__ == "__main__":
    main()
