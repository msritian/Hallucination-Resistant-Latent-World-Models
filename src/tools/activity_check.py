import glob, torch
from src.tools import audit_ablations as aa
from src.preflight.metrics import stratified_auroc
dumps = aa.load_dumps(sorted(glob.glob('/tmp/feat/feat_*.tar.gz')))

def activity(part, L):
    v, r = part["v_hat"][:L].float(), part["r_hat"][:L].float()
    a = torch.zeros_like(v); a[1:] = (v[1:] - v[:-1]).abs() + (r[1:] - r[:-1]).abs()
    return a

def feats(part, L, mode):
    X = aa.features(part, aa.CRITIC, True, L)                    # [L, M, F] baseline features
    a = activity(part, L)
    if mode == "base": return X
    if mode == "act":                                              # trees also see how busy each step is
        return torch.cat([X, a.unsqueeze(-1), torch.cummax(a, 0).values.unsqueeze(-1)], -1)
    if mode == "norm":                                             # residual signals divided by activity
        cols = []
        for k in aa.CRITIC:
            x = part["signals"][k][:L].float() / (a + a.median() + 1e-6)
            cols += [x, torch.cummax(x, 0).values]
        return torch.cat([X, torch.stack(cols, -1)], -1)

def run(mode):
    res = {"all": [], "gradual": [], "sudden": []}; busy_fa = []; calm_fa = []
    for r, d in dumps.items():
        cal, ev = d["cal"], d["ev"]; L = cal["E"].shape[0]
        pos, keep, _, _ = aa.labels(cal, cal["E"])
        Xc = feats(cal, L, mode); F = Xc.shape[-1]
        m = aa.gbt().fit(Xc.reshape(-1, F)[keep.reshape(-1)].numpy(), pos.reshape(-1)[keep.reshape(-1)].numpy())
        Xe = feats(ev, L, mode)
        p = torch.as_tensor(m.predict_proba(Xe.reshape(-1, F).numpy())[:, 1]).view(L, -1).float()
        a = aa.aurocs(p, d)
        for c in res: res[c].append(a[c])
        # false alarms on CORRECT steps: share flagged (top 25% of p) among busy vs calm correct steps
        epos, ekeep, _, _ = aa.labels(ev, cal["E"]); neg = ekeep & ~epos
        thr = torch.quantile(p.flatten(), 0.75); act = activity(ev, L)
        q1, q2 = torch.quantile(act[neg], 0.33), torch.quantile(act[neg], 0.67)
        flag = p > thr
        busy_fa.append(float(flag[neg & (act > q2)].float().mean())); calm_fa.append(float(flag[neg & (act <= q1)].float().mean()))
    mean = lambda v: sum(v) / len(v)
    print(f"{mode:5s}  AUROC all/gradual/sudden: {mean(res['all']):.3f} / {mean(res['gradual']):.3f} / {mean(res['sudden']):.3f}"
          f"   false alarms on correct steps: busy {mean(busy_fa):.3f}  calm {mean(calm_fa):.3f}")
for mode in ("base", "act", "norm"): run(mode)
