"""Write the LATE EMA weights of a training checkpoint as a finished flow file.

train_lpd ships the weights its validation liked best. For this run that is step 1199, while
the training objective -- the thing actually being optimised -- was still descending and hit
its best quintile around step 1599. Validation here scores 16 held-out bodies at one fixed
timestep each, so it cannot resolve that; selecting on it discards the late weights.

load_flow_file on a raw .ckpt returns best_state, i.e. the validation-selected step, so
pointing a reconstruction at the checkpoint does NOT give the late weights. This rebuilds the
EMA average explicitly -- shadow / (1 - decay**n), exactly as EMA.state does -- and writes it
in the {"state_dict", "meta"} layout a finished file uses, so check_flow_metadata still sees
the corpus, prior, phase grid and operator the weights were trained under.

usage: extract_late.py CKPT OUT [DECAY]
"""
import sys
from pathlib import Path

import torch

META_KEYS = ("corpus", "prior", "phases", "operator_res", "render", "bodies", "dim",
             "n_experts", "n_modes", "fit_from", "fit_weight", "occ_eps", "occ_weight",
             "train_geoms", "n_val", "loss")


def main() -> None:
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    decay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99   # as printed in the run's log

    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net, shadow, n = st["net"], st.get("ema"), int(st.get("ema_n", 0))
    if not shadow or n == 0:
        raise SystemExit(f"{ckpt_path} carries no EMA shadow; nothing late to extract")

    c = 1.0 - decay ** n
    out = {}
    for k, v in net.items():
        out[k] = (shadow[k] / c).to(v.dtype) if k in shadow else v.detach().clone()

    meta = {k: st[k] for k in META_KEYS if k in st}
    meta.update({"steps_trained": int(st["step"]), "best_step": int(st["step"]),
                 "val": None,
                 "selected_by": ("late EMA over the last ~%d steps, not the validation-best "
                                 "step (%s at val %s)" % (round(1 / (1 - decay)),
                                                          st.get("best_step"), st.get("best")))})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out, "meta": meta}, out_path)

    same = sum(1 for k in out if k in shadow)
    print(f"  checkpoint step {st['step']}, EMA n={n}, decay={decay}, bias correction {c:.6f}")
    print(f"  validation-selected step would have been {st.get('best_step')} "
          f"(val {st.get('best')})")
    print(f"  wrote {out_path}: {same}/{len(out)} tensors from the EMA shadow")


if __name__ == "__main__":
    main()
