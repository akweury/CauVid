"""What if the rules read the slots before an agent appeared as unknown, or as true? One event (2026-11-27).

A slot is "known" if the video's history covers it (`TemporalRuleNet.layer_inputs`), whether or not the agent was
there: for an agent seen for one frame, its slots 2 / 3 are known with no atom, so "X@2" is false and "!X@2" TRUE
("it did not stop 2 s ago" for a car that was not there). Settings of the rule model's score of one row (video,
agent, target frame t):
  trained     as the rules were fit and evaluated (absent = known, no atom);
  absent->u   the slots where the agent held no atom (it was not in view; own and role scopes) are unknown, and
              every unknown base literal (positive or negated) gets the value u (--ev_unknown, default 0 and 1);
              derived literals are recomputed from them.
Scorings of the same fitted rules (no refit; 2026-11-27, the user: a rule that does not hold must give 0):
  soft        as fit, a_r = prod (1 - g (1 - x)): a rule with a false literal still gives prod (1 - g) of its weight;
  hard_kept   a_r = the AND of the literals with gate >= --ev_gate_cut (0 / 1), the others dropped;
  hard_full   a_r = the AND of the full mined body (every gate 1).
Per setting and scoring: P vs the rules' threshold and the logit = bias + positive rules + negative rules; at the target
frame the rules changing most and the largest contributions. Rules: the run's rule-lab .rules.pt files.

    ./run_roadpp_docker.sh --gpu 3 --event_whatif -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --ev_video sumo_00473 --ev_agent 19 --ev_frame 57 --ev_head Car:Jun --ev_span 10
--ev_span k: also the next k frames of the agent (one line each). Prints only; also <output_dir>/rule_lab/whatif_*.json.
"""
import glob

import torch

from src.exp_roadpp import config, rule_lab, rule_paper, step05, step05_mine, utils_data, utils_pipe


def literal_bits(model, derived, inputs, u, absent_unknown=False):
    """(B, L + D + 1) float literals with the unknown base literals set to u, the derived ones recomputed from them
    (with u in (0, 1): between the two hard settings), plus the true one. absent_unknown: the own / role scope
    inputs of slots where the agent held no atom are unknown (not "known absent")."""
    x, k = model.aligned_inputs(inputs)
    if absent_unknown:
        S, J, A = model.num_scopes, model.num_memory_slots, model.num_atoms
        x3, k3 = x.view(-1, S, J, A), k.view(-1, S, J, A).clone()
        absent = x3[:, 0].sum(dim=-1) == 0                                    # (B, J) the agent not in view
        for s in [0] + list(range(2, S)):                                     # own and role scopes, not "other"
            k3[:, s][absent] = 0
        k = k3.view(len(x), -1)
    true, false = model.literal_truth(x, k)
    unknown = (1 - true - false).clamp(min=0)
    base = true + u * unknown
    if u in (0.0, 1.0):
        ext = derived.extend(base > 0.5).float()
    else:
        # derived literals: AND of base literals -> soft value as the min over the two hard settings' mean
        lo, hi = derived.extend(true > 0.5).float(), derived.extend((true + unknown) > 0.5).float()
        ext = torch.cat([base, (lo + u * (hi - lo))[:, base.shape[1]:]], dim=1)
    return torch.cat([ext, torch.ones((len(ext), 1), device=ext.device)], dim=1), unknown


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--ev_video", type=str, required=True)
    parser.add_argument("--ev_agent", type=int, required=True, help="the agent index n of the video tensor")
    parser.add_argument("--ev_frame", type=int, required=True, help="the target frame t (the row's frame)")
    parser.add_argument("--ev_head", type=str, required=True)
    parser.add_argument("--ev_span", type=int, default=0, help="also the next k frames")
    parser.add_argument("--ev_unknown", type=str, default="0,1", help="values of the unknown literals")
    parser.add_argument("--ev_thr", type=float, default=0.403, help="the rules' threshold (former_clips: 2% FPR)")
    parser.add_argument("--ev_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--ev_top", type=int, default=8)
    parser.add_argument("--ev_gate_cut", type=float, default=0.5, help="hard_kept: literals with gate >= this kept")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    derived = rule_lab.Derived(model, names)
    ext = rule_lab.ExtModel(model, derived, True)
    text = lambda m: step05_mine.literal_text(ext, names, m)
    files = sorted(glob.glob(str(lab_dir / f"{args.ev_rules}.rules.pt")))
    sc = rule_paper.RuleScorer(files, model, derived, 1, device)
    h = names.index(args.ev_head)
    sel = torch.nonzero(sc.head == h).flatten()
    body, gate, weight, bias = sc.body[sel], sc.gate[sel], sc.weight[sel], float(sc.bias[h])
    pad = model.num_literals + derived.size                       # the always-true padding literal (index L)
    # settings: (name, u, absent_unknown); the first is the reference of the changes
    gates = {"soft": gate, "hard_kept": (gate >= args.ev_gate_cut).float(), "hard_full": torch.ones_like(gate)}
    us = [("trained", 0.0, False)] + [(f"absent->{float(u):g}", float(u), True) for u in args.ev_unknown.split(",")]

    def rule_str(r):
        return " & ".join(f"{text(m)}" + (f" (g {g:.2f})" if g < 0.95 else "")
                          for m, g in zip(body[r].tolist(), gate[r].tolist()) if m < pad and g >= rule_lab.DROP_GATE)

    frames = torch.arange(args.ev_frame, args.ev_frame + args.ev_span + 1)
    with torch.no_grad():
        rows = step05._video_rows(model, training_data, args.ev_video, frames, device)
        pick = torch.nonzero(rows["rows"][:, 1] == args.ev_agent).flatten()
        if not len(pick):
            raise ValueError(f"agent {args.ev_agent} has no row in frames {frames.tolist()} of {args.ev_video}")
        out = {"video": args.ev_video, "agent": args.ev_agent, "head": args.ev_head, "bias": bias, "frames": []}
        print(f"\n[What if] {args.ev_video} agent {args.ev_agent} {args.ev_head}: {len(sel)} rules, bias {bias:+.2f}, "
              f"rules' threshold {args.ev_thr:.3f}")
        for i in pick.tolist():
            t = int(rows["rows"][i, 0])
            first = int(rows["first"][i, h])
            inputs = rows["inputs"][i:i + 1]
            res, acts = {}, {}
            for (u0, uval, ab), (sg, gt) in [(x, y) for x in us for y in gates.items()]:
                u = u0 if sg == "soft" else f"{u0}/{sg}"
                lits, unknown = literal_bits(model, derived, inputs, uval, ab)
                a = rule_lab._soft_act(lits, body, gt)[0]
                c = weight * a
                z = bias + float(c.sum())
                res[u] = {"p": float(torch.sigmoid(torch.tensor(z))), "logit": z, "positive": float(c[c > 0].sum()),
                          "negative": float(c[c < 0].sum()), "active_rules": int((a >= 0.5).sum())}
                acts[u] = a
            n_unknown = int((unknown[0] > 0.5).sum())
            print(f"\n t={t} (onset in {first} frames; {n_unknown} base literals unknown when absent = unknown)")
            for u, r in res.items():
                print(f"   {u:<22} P {r['p']:.3f} {'>=' if r['p'] >= args.ev_thr else '< '} thr {args.ev_thr:.3f}   "
                      f"logit = bias {bias:+.2f} + pos {r['positive']:+.2f} + neg {r['negative']:+.2f} = {r['logit']:+.2f}"
                      f"   ({r['active_rules']} rules active)")
            fr = {"t": t, "onset_in": first, "unknown_literals": n_unknown, "scores": res}
            if t == args.ev_frame:
                lits0, unk = literal_bits(model, derived, inputs, 0.0)
                absent = [j + 1 for j in range(model.num_memory_slots)
                          if float(model.aligned_inputs(inputs)[0].view(model.num_scopes, model.num_memory_slots, -1)[0, j].sum()) == 0]
                print(f"   slots where the agent was not in view: {absent or 'none'}")
                own_true = [text(m) for m in torch.nonzero(lits0[0, :model.num_literals] > 0.5).flatten().tolist()
                            if "(own)" in text(m) and not text(m).startswith("!")]
                print(f"   own literals true: {', '.join(own_true)}")
                ref = us[0][0]
                for u, _uv, _ab in us[1:]:
                    d = weight * (acts[u] - acts[ref])
                    order = torch.argsort(d.abs(), descending=True)[:args.ev_top].tolist()
                    print(f"   rules changing most, {ref} -> {u} (w x activation change):")
                    for r in order:
                        print(f"     {float(d[r]):+.2f}  w {float(weight[r]):+.2f}  act {float(acts[ref][r]):.2f} -> "
                              f"{float(acts[u][r]):.2f}  {rule_str(r)}")
                    fr[f"changes_{u}"] = [{"delta": float(d[r]), "w": float(weight[r]),
                                           "act": [float(acts[ref][r]), float(acts[u][r])],
                                           "rule": rule_str(r)} for r in order]
                for u in [x[0] for x in us] + [f"trained/{sg}" for sg in gates if sg != "soft"]:
                    c = weight * acts[u]
                    k = args.ev_top if "/" in u else 4
                    top = torch.argsort(c, descending=True)[:k].tolist() + torch.argsort(c)[:k].tolist()
                    print(f"   largest contributions, {u}:")
                    for r in top:
                        if abs(float(c[r])) >= 0.05:
                            print(f"     {float(c[r]):+.2f}  w {float(weight[r]):+.2f} act {float(acts[u][r]):.2f}  {rule_str(r)}")
            out["frames"].append(fr)
    path = lab_dir / f"whatif_{args.ev_video}_a{args.ev_agent}_f{args.ev_frame}_{args.ev_head.replace(':', '_')}.json"
    utils_data.save_json(out, path)
    print(f"\n[What if] {path}")


if __name__ == "__main__":
    main()
