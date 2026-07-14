"""Qualitative inference probe — generates from a fixed prompt set across
multiple task families. Output is a JSON of (prompt, generation) pairs for
human inspection, not a numeric eval.

Use to compare capability footprints between models (vanilla d=2048 s4 vs
PCC d=2048 reason etc.) and propose next experiments based on what each
model can/can't do.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


PROBES = [
    # === Chain in-distribution (sanity) ===
    ("chain_id_k3",
     "f(0)=3 f(1)=7 f(2)=2 f(3)=5 f(4)=1 f(5)=4 f(6)=8 f(7)=2 f(8)=11 f(9)=0 f(10)=6 f(11)=9. f^3(0) = "),
    ("chain_id_k7",
     "f(0)=3 f(1)=7 f(2)=2 f(3)=5 f(4)=1 f(5)=4 f(6)=8 f(7)=2 f(8)=11 f(9)=0 f(10)=6 f(11)=9. f^7(0) = "),
    # === Chain OOD (k>8 — extrapolation depth) ===
    ("chain_ood_k12",
     "f(0)=3 f(1)=7 f(2)=2 f(3)=5 f(4)=1 f(5)=4 f(6)=8 f(7)=2 f(8)=11 f(9)=0 f(10)=6 f(11)=9. f^12(0) = "),
    ("chain_ood_k16",
     "f(0)=3 f(1)=7 f(2)=2 f(3)=5 f(4)=1 f(5)=4 f(6)=8 f(7)=2 f(8)=11 f(9)=0 f(10)=6 f(11)=9. f^16(0) = "),
    # === Listops in-dist + OOD depth ===
    ("listops_id_d2", "Compute: MIN[3,7,1,9] = "),
    ("listops_id_d3", "Compute: MAX[MIN[5,2],MAX[3,1]] = "),
    ("listops_ood_d5",
     "Compute: MIN[MAX[MIN[3,7],MED[5,2,8]],MAX[1,MIN[4,6,2],9]] = "),
    # === Modular in-dist + OOD prime ===
    ("modular_id_p7", "Compute: (8 + 9) mod 7 = "),
    ("modular_id_p13", "Compute: (15 + 6) mod 13 = "),
    ("modular_ood_p23", "Compute: (15 + 8) mod 23 = "),
    ("modular_ood_p29", "Compute: (22 + 11) mod 29 = "),
    # === Math arithmetic (basic) ===
    ("arith_add", "12 + 34 = "),
    ("arith_mul", "7 * 8 = "),
    ("arith_sub", "100 - 47 = "),
    ("arith_div", "144 / 12 = "),
    # === Math word problem (NuminaMath-like) ===
    ("math_word_speed",
     "If a train travels 60 mph for 2 hours, how far does it go? Answer: "),
    ("math_word_solve", "Solve for x: 2x + 3 = 11. x = "),
    ("math_word_perim",
     "A rectangle has length 5 and width 3. What is its perimeter? Answer: "),
    # === Logic / deduction ===
    ("logic_modus_ponens",
     "All birds can fly. A sparrow is a bird. Therefore a sparrow can "),
    ("logic_transitive",
     "Alice is taller than Bob. Bob is taller than Carol. Who is the tallest? Answer: "),
    ("logic_negation",
     "If it rains, the ground gets wet. The ground is dry. Did it rain? Answer: "),
    # === General knowledge ===
    ("gen_capital", "The capital of France is "),
    ("gen_planet", "The largest planet in the solar system is "),
    ("gen_boil", "Water boils at "),
    ("gen_color", "The sky is the color "),
    # === Code (Python) ===
    ("code_add", "def add(a, b):\n    return "),
    ("code_factorial",
     "def factorial(n):\n    if n == 0:\n        return 1\n    return n * "),
    # === Story / continuation (LM signal) ===
    ("story_open",
     "Once upon a time, there was a wizard who "),
    ("story_dialogue",
     '"Where are we?" asked Alice. "We are in '),
]


def _load_tokenizer(data_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(data_dir / "tokenizer"))


def _load_model(ckpt_dir: Path, device: str):
    state_path = ckpt_dir / "state.pt"
    cfg_path = ckpt_dir.parent.parent / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    from pretrain.model import PretrainConfig, build_model
    arch_cfg_dict = cfg_data["arch"]
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    arch_cfg = PretrainConfig(**{k: v for k, v in arch_cfg_dict.items()
                                  if k in field_names})
    model = build_model(arch_cfg).to(device)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, arch_cfg, cfg_data


@torch.no_grad()
def _greedy(model, tokenizer, prompt: str, n_loops: int, device, dtype,
             max_new_tokens: int = 80) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type=device, dtype=dtype,
                                  enabled=(dtype != torch.float32)):
            out = model(ids, n_loops=n_loops)
        nxt = int(out["logits"][0, -1, :].argmax().item())
        generated.append(nxt)
        if nxt == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if ids.size(1) >= model.cfg.max_seq_len:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


def run(ckpt_dir: str, data_dir: str, out_path: str = "probe_eval.json",
        inference_n_loops: int | None = None,
        max_new_tokens: int = 80) -> dict:
    ckpt_path = Path(ckpt_dir)
    data_path = Path(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda"
                                and torch.cuda.get_device_capability(0)[0] >= 8) \
              else (torch.float16 if device == "cuda" else torch.float32)
    print(f"device={device} dtype={dtype}", flush=True)

    model, arch_cfg, cfg_data = _load_model(ckpt_path, device)
    n_loops = inference_n_loops if inference_n_loops is not None else arch_cfg.n_loops
    print(f"arch={arch_cfg.arch}  d={arch_cfg.d_model}  trained_n_loops={arch_cfg.n_loops}  "
          f"inference_n_loops={n_loops}", flush=True)
    tokenizer = _load_tokenizer(data_path)

    results: list[dict] = []
    t0 = time.time()
    for i, (name, prompt) in enumerate(PROBES):
        gen = _greedy(model, tokenizer, prompt, n_loops, device, dtype, max_new_tokens)
        results.append({"name": name, "prompt": prompt, "gen": gen})
        print(f"  [{i+1:>2}/{len(PROBES)}] {name}", flush=True)
        print(f"    PROMPT: {prompt!r}", flush=True)
        print(f"    GEN:    {gen!r}", flush=True)
        print(flush=True)

    payload = {
        "ckpt_dir": str(ckpt_path),
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "trained_n_loops": arch_cfg.n_loops,
        "n_loops": n_loops,
        "max_new_tokens": max_new_tokens,
        "wall_sec": time.time() - t0,
        "results": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out_path}", flush=True)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="probe_eval.json")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, out_path=args.out,
        max_new_tokens=args.max_new_tokens)
