
import os, argparse, yaml, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output_dir"]
    df = pd.read_parquet(os.path.join(out_dir, "metrics.parquet"))

    for metric in ["NMSE", "absorption_rate", "split_improve_2-1", "dead_rate"]:
        for variant, d in df.groupby("variant"):
            fig = plt.figure()
            for key, g in d.groupby("sparsity"):
                g2 = g.sort_values("step")
                plt.plot(g2["step"], g2[metric], marker="o", label=f"s{key}")
            plt.xlabel("step"); plt.ylabel(metric); plt.title(f"{metric} vs step ({variant})")
            plt.legend(); fig.savefig(os.path.join(out_dir, f"fig_{metric}_vs_step_{variant}.png"), bbox_inches="tight")
            plt.close(fig)

    if "dense_rate@0.2" in df.columns:
        for variant, d in df.groupby("variant"):
            fig = plt.figure()
            for key, g in d.groupby("sparsity"):
                g2 = g.sort_values("step")
                plt.plot(g2["step"], g2["dense_rate@0.2"], marker="o", label=f"dense0.2 s{key}")
            plt.xlabel("step"); plt.ylabel("dense_rate@0.2"); plt.title(f"Dense(0.2) vs step ({variant})")
            plt.legend(); fig.savefig(os.path.join(out_dir, f"fig_dense0.2_vs_step_{variant}.png"), bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    main()
