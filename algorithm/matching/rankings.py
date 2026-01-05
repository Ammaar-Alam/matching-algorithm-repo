from __future__ import annotations

import argparse, json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.20)
    ap.add_argument("--out", default="lists.json")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    # build for each A a list of B ranked by score
    lists = {}
    for a, g in df.groupby("A"):
        # drop very low LCB candidates if present
        if "LCB_A" in g.columns:
            g = g[g["LCB_A"] >= args.tau]
        gg = g.sort_values(by=["score"], ascending=False)
        lists[str(a)] = [str(b) for b in gg["B"].tolist()]

    with open(args.out, "w") as f:
        json.dump(lists, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

