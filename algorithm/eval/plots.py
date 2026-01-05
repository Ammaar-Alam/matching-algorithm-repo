from __future__ import annotations

import argparse, json, os


def plot_or_dump_domain_weights(weights_json: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    with open(weights_json, 'r', encoding='utf-8') as f:
        w = json.load(f)
    # filter to domain keys
    domains = [k for k in w.keys() if k not in ("cv_auc",)]
    vals = [float(w[k]) for k in domains]
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        xs = range(len(domains))
        plt.bar(xs, vals)
        plt.xticks(xs, domains, rotation=30, ha='right')
        plt.ylabel('weight')
        plt.tight_layout()
        p = os.path.join(outdir, 'domain_weights.png')
        plt.savefig(p, dpi=160)
        print(f'wrote {p}')
    except Exception:
        # text fallback
        p = os.path.join(outdir, 'domain_weights.txt')
        with open(p, 'w') as f:
            for k,v in zip(domains, vals):
                f.write(f'{k}\t{v}\n')
        print(f'wrote {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_json', required=True)
    ap.add_argument('--outdir', default='interp')
    args = ap.parse_args()
    plot_or_dump_domain_weights(args.weights_json, args.outdir)


if __name__ == '__main__':
    main()

