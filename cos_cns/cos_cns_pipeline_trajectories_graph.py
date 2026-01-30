#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COS-CNS FAST GRAPH pipeline (English) (multi-topology, COS-NUM/COS-STAB outputs, with progress display)

Main features:
- chain / star / Erdos-Renyi (összefüggő) topológián futtatható referencia instanciálás
- no-signaling (TVD), strict cone (távolság-sáv heatmap), scheduling (confluent vs nonconfluent), NC1–NC4
- COS-NUM: runs/<run_id>/config.json + outputs/*.csv/*.npz + outputs/summary.json
- COS-STAB: logs/timeseries.csv
- LaTeX kanonikus ábrák: csak a publish_topology-hoz másolódnak ki (figs/*.pdf + fig-nc-*.pdf)

Note: this script does not rely on Python's hash() for seeds, so PYTHONHASHSEED is not required for reproducibility.

Example (Windows CMD, single line):
  python cos_cns_pipeline_fast_graph_multi.py --topology_list chain,star,er --publish_topology er --N 15 --steps 12 --trials 60 --ntraj 1200 --seed 42

Progress:
  Enabled by default. Disable: --no_progress
  Refresh interval: --progress_interval 0.2
"""
import argparse, csv, json
import sys, time
from dataclasses import asdict, dataclass
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
})

# --- Single-qubit ops ---
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)

# CNOT (control first, target second) in |00>,|01>,|10>,|11>
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]], dtype=complex)


# 2-qubit gates (symmetric options for undirected graphs)
SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]], dtype=complex)

# sqrt(SWAP) (entangling + excitation mixing)
sqrtSWAP = np.array([[1,0,0,0],
                     [0,(1+1j)/2,(1-1j)/2,0],
                     [0,(1-1j)/2,(1+1j)/2,0],
                     [0,0,0,1]], dtype=complex)

# Hadamard (NC4-hez: superpozíció)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
# ---------- Progress ----------
def _fmt_seconds(sec: float) -> str:
    if sec is None or sec != sec or sec < 0:
        return "?"
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

class SimpleProgress:
    def __init__(self, total: int, prefix: str, enabled: bool, min_interval: float):
        self.total = int(max(0, total))
        self.prefix = prefix
        self.enabled = enabled
        self.min_interval = float(min_interval)
        self.t0 = time.time()
        self.last = 0.0

    def update(self, i: int, extra: str = ""):
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < self.min_interval and i < self.total:
            return
        self.last = now
        i = max(0, min(int(i), self.total))
        frac = (i / self.total) if self.total > 0 else 1.0
        bar_len = 28
        filled = int(round(bar_len * frac))
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = now - self.t0
        eta = (elapsed * (1 - frac) / frac) if frac > 1e-9 else None
        msg = f"{self.prefix} [{bar}] {i}/{self.total}  elapsed={_fmt_seconds(elapsed)}  eta={_fmt_seconds(eta)}"
        if extra:
            msg += f"  {extra}"
        print("\r" + msg + " " * 4, end="", file=sys.stderr, flush=True)
        if i >= self.total:
            print("", file=sys.stderr, flush=True)

# ---------- Utilities ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def tvd(p,q): return 0.5*np.sum(np.abs(p-q))

def now_run_prefix(rng: np.random.Generator):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rnd = "".join(rng.choice(list("0123456789abcdef"), size=6))
    return f"{ts}_{rnd}"

# ---------- Graph helpers ----------
def build_chain(n): return [(i,i+1) for i in range(n-1)]
def build_star(n): return [(0,i) for i in range(1,n)]

def neighbors(n, edges):
    adj=[[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    return adj

def is_connected(n, edges):
    if n<=1: return True
    adj=neighbors(n,edges)
    seen=[False]*n
    stack=[0]; seen[0]=True
    while stack:
        u=stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v]=True; stack.append(v)
    return all(seen)

def build_er_connected(n,p,rng,max_tries=2000):
    for _ in range(max_tries):
        edges=[]
        for i in range(n):
            for j in range(i+1,n):
                if rng.random()<p:
                    edges.append((i,j))
        if is_connected(n,edges):
            return edges
    return build_chain(n)

def bfs_dist(n, edges, src):
    adj=neighbors(n,edges)
    dist=[-1]*n
    dist[src]=0
    q=[src]; qi=0
    while qi<len(q):
        u=q[qi]; qi+=1
        for v in adj[u]:
            if dist[v]<0:
                dist[v]=dist[u]+1
                q.append(v)
    return dist

def approx_diameter_endpoints(n, edges, rng):
    start=int(rng.integers(0,n))
    d0=bfs_dist(n,edges,start); a=int(np.argmax(d0))
    d1=bfs_dist(n,edges,a); b=int(np.argmax(d1))
    return a,b,int(d1[b])

def greedy_matchings(edges, rng):
    E=edges.copy()
    rng.shuffle(E)
    remaining=E
    matchings=[]
    while remaining:
        used=set()
        M=[]
        new=[]
        for u,v in remaining:
            if u not in used and v not in used:
                M.append((u,v))
                used.add(u); used.add(v)
            else:
                new.append((u,v))
        matchings.append(M)
        remaining=new
    return matchings

# ---------- Efficient statevector gates ----------
def apply_1q_gate(psi, gate, N, q):
    psi_t=psi.reshape([2]*N)
    psi_t=np.moveaxis(psi_t,q,0)
    psi_t=(gate @ psi_t.reshape(2,-1)).reshape(2,*psi_t.shape[1:])
    psi_t=np.moveaxis(psi_t,0,q)
    return psi_t.reshape(-1)

def apply_2q_gate(psi, gate4, N, q1, q2):
    assert q1!=q2
    a,b=(q1,q2) if q1<q2 else (q2,q1)
    psi_t=psi.reshape([2]*N)
    psi_t=np.moveaxis(psi_t,[a,b],[0,1])
    flat=psi_t.reshape(4,-1)
    flat=gate4 @ flat
    psi_t=flat.reshape(2,2,*psi_t.shape[2:])
    psi_t=np.moveaxis(psi_t,[0,1],[a,b])
    return psi_t.reshape(-1)

def amp_damp_step(psi,N,q,gamma,rng):
    g=float(gamma)
    if g<=0: return psi
    psi_t=psi.reshape([2]*N)
    psi_t=np.moveaxis(psi_t,q,0)
    a0=psi_t[0].reshape(-1); a1=psi_t[1].reshape(-1)
    p_jump=g*float(np.vdot(a1,a1).real)
    if rng.random()<p_jump and p_jump>0:
        a0_new=np.sqrt(g)*a1; a1_new=np.zeros_like(a1); norm=np.sqrt(p_jump)
    else:
        a0_new=a0; a1_new=np.sqrt(1-g)*a1; norm=np.sqrt(max(1-p_jump,1e-16))
    psi_new=np.stack([a0_new,a1_new],axis=0).reshape(2,*psi_t.shape[1:])
    psi_new=np.moveaxis(psi_new,0,q).reshape(-1)
    psi_new/=norm
    return psi_new

def z_probs_site(psi,N,q):
    dim=psi.size; mask=1<<(N-1-q)
    probs=np.abs(psi)**2
    p1=float(np.sum(probs[(np.arange(dim)&mask)!=0])); p0=1.0-p1
    p0=max(0.0,min(1.0,p0)); p1=max(0.0,min(1.0,p1))
    return np.array([p0,p1],dtype=float)

def z_expect_site(psi,N,q):
    p=z_probs_site(psi,N,q); return float(p[0]-p[1])

def init_state_zero(N):
    psi=np.zeros((2**N,),dtype=complex); psi[0]=1.0; return psi


def init_state_superposition(N):
    """|+...+> inicializálás: Hadamard minden qubitre.
    Scheduling/NC3 esetén ez tipikusan nemtriviális Z-metrikát ad.
    """
    psi = init_state_zero(N)
    for q in range(N):
        psi = apply_1q_gate(psi, H, N, q)
    return psi

@dataclass
class Params:
    N:int=12
    gamma:float=0.05
    seed:int=42
    topology:str="chain"
    p:float=0.4

class GraphDynamics:
    """One time step = one matching layer + damping. Strict cone speed: 1 per step."""
    def __init__(self, p: Params, edges, matchings, edge_gate):
        self.p=p; self.edges=edges; self.matchings=matchings; self.L=len(matchings)
        self.edge_gate = edge_gate

    def step(self, psi, rng, t, permute_within):
        M=self.matchings[t%self.L].copy()
        if permute_within:
            M = list(reversed(M))  # determinisztikus permutáció; nem fogyaszt RNG-t
        for u,v in M:
            a,b=(u,v) if u<v else (v,u)
            psi=apply_2q_gate(psi,self.edge_gate,self.p.N,a,b)
        for q in range(self.p.N):
            psi=amp_damp_step(psi,self.p.N,q,self.p.gamma,rng)
        return psi

    def step_nonconfluent(self, psi, rng):
        E=self.edges.copy(); rng.shuffle(E)
        for u,v in E:
            a,b=(u,v) if u<v else (v,u)
            psi=apply_2q_gate(psi,self.edge_gate,self.p.N,a,b)
        for q in range(self.p.N):
            psi=amp_damp_step(psi,self.p.N,q,self.p.gamma,rng)
        return psi

def nonlocal_filter_like(psi,p,rng,A,B):
    # project A in Z; if outcome=1, apply X on B
    pA=z_probs_site(psi,p.N,A); u=rng.random()
    if u<pA[1]:
        psi=apply_1q_gate(psi,P1,p.N,A); psi/=np.sqrt(max(pA[1],1e-16))
        psi=apply_1q_gate(psi,X,p.N,B)
    else:
        psi=apply_1q_gate(psi,P0,p.N,A); psi/=np.sqrt(max(pA[0],1e-16))
    return psi

def write_csv(path: Path, rows, fieldnames):
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fieldnames})

def write_json(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def print_run_table(rows):
    """
    Print a compact result table to stdout.
    rows: list of dicts with keys: topology, run_id, dist_ab, num_edges, num_matchings, pass_ns, pass_sched, overall
    """
    if not rows:
        return
    # Compute column widths
    headers = ["topology", "run_id", "dist(A,B)", "|E|", "matchings", "pass(ns)", "pass(sched)", "overall"]
    data = []
    for r in rows:
        data.append([
            r["topology"],
            r["run_id"],
            str(r["dist_ab"]),
            str(r["num_edges"]),
            str(r["num_matchings"]),
            "OK" if r["pass_ns"] else "FAIL",
            "OK" if r["pass_sched"] else "FAIL",
            "OK" if r["overall"] else "FAIL",
        ])
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt_row(row):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
    print("\nResults summary (per topology):")
    print(fmt_row(headers))
    print("-" * (sum(widths) + 2*(len(widths)-1)))
    for row in data:
        print(fmt_row(row))
    print()

def write_index_json(prefix_dir: Path, rows, args_dict):
    """
    Write runs/<prefix>/index.json that aggregates per-topology results.
    """
    index = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_args": args_dict,
        "runs": rows,
    }
    ensure_dir(prefix_dir)
    write_json(prefix_dir / "index.json", index)


# ---------- Experiments (with progress) ----------
def exp_nosignal(dyn,p,steps,ntraj,A,B, prog: SimpleProgress|None):
    rows=[]; psi0=init_state_zero(p.N); psi1=apply_1q_gate(psi0,X,p.N,A)
    for t in range(steps+1):
        ps0=[]; ps1=[]
        base = t*ntraj
        for k in range(ntraj):
            if prog: prog.update(base+k+1, extra=f"t={t}/{steps}")
            rng=np.random.default_rng(p.seed+10000+97*t+k)
            rng2=np.random.default_rng(); rng2.bit_generator.state=rng.bit_generator.state.copy()
            a=psi0.copy(); b=psi1.copy()
            for s in range(t):
                a=dyn.step(a,rng,s,False); b=dyn.step(b,rng2,s,False)
            ps0.append(z_probs_site(a,p.N,B)); ps1.append(z_probs_site(b,p.N,B))
        p0m=np.mean(ps0,axis=0); p1m=np.mean(ps1,axis=0)
        rows.append({"t":t,"p0_0":float(p0m[0]),"p0_1":float(p0m[1]),
                     "p1_0":float(p1m[0]),"p1_1":float(p1m[1]),
                     "tvd":float(tvd(p0m,p1m)),"delta_fail":0.0,"ntraj":ntraj})
    return rows

def exp_cone(dyn,p,steps,ntraj,center,distC, prog: SimpleProgress|None):
    maxd=max(distC); infl=np.zeros((steps+1,maxd+1),dtype=float)
    psi0=init_state_zero(p.N); psi1=apply_1q_gate(psi0,X,p.N,center)
    for t in range(steps+1):
        vals=np.zeros((maxd+1,),dtype=float); cnt=np.zeros((maxd+1,),dtype=int)
        base = t*ntraj
        for k in range(ntraj):
            if prog: prog.update(base+k+1, extra=f"t={t}/{steps}")
            rng=np.random.default_rng(p.seed+20000+97*t+k)
            rng2=np.random.default_rng(); rng2.bit_generator.state=rng.bit_generator.state.copy()
            a=psi0.copy(); b=psi1.copy()
            for s in range(t):
                a=dyn.step(a,rng,s,False); b=dyn.step(b,rng2,s,False)
            for x in range(p.N):
                d=distC[x]
                vals[d]+=abs(z_expect_site(a,p.N,x)-z_expect_site(b,p.N,x))
                cnt[d]+=1
        infl[t,:]=vals/np.maximum(cnt,1)
    return infl


def exp_sched(dyn, p, steps, trials, ntraj, B, mode, prog: Optional[SimpleProgress]):
    """
    Scheduling benchmark / NC3 alap:
    - reference: fix (permute_within=False)
    - variant:
        mode == "confluent"    -> permute_within=True (csak diszjunkt éleken belüli permutáció)
        mode == "nonconfluent" -> step_nonconfluent (átfedő élek véletlen sorrendben)

    Zajcsökkentés: minden trajektóriára közös véletlenforrást használunk (rng és rng2 azonos állapotból),
    így a konfluens esetben a különbség ideally 0 (numerikus kerekítésen belül).
    """
    psi0 = init_state_superposition(p.N)  # nemtriviális kezdőállapot Bob Z-TVD-hez
    rows = []
    pref_list = []
    for i in range(trials):
        done = i * ntraj
        ps_ref = []
        ps_var = []
        for k in range(ntraj):
            if prog:
                prog.update(done + k + 1, extra=f"trial={i+1}/{trials}")
            rng = np.random.default_rng(p.seed + 40000 + 100*i + k)
            rng2 = np.random.default_rng()
            rng2.bit_generator.state = rng.bit_generator.state.copy()

            a = psi0.copy()
            b = psi0.copy()
            for s in range(steps):
                a = dyn.step(a, rng, s, permute_within=False)
                if mode == "confluent":
                    b = dyn.step(b, rng2, s, permute_within=True)
                else:
                    b = dyn.step_nonconfluent(b, rng2)

            ps_ref.append(z_probs_site(a, p.N, B))
            ps_var.append(z_probs_site(b, p.N, B))

        pref = np.mean(ps_ref, axis=0)
        pvar = np.mean(ps_var, axis=0)
        pref_list.append(pref)

        rows.append({
            "sample": i,
            "p0": float(pvar[0]),
            "p1": float(pvar[1]),
            "tvd_to_ref": float(tvd(pref, pvar)),
            "delta_fail_to_ref": 0.0,
            "ntraj": int(ntraj),
        })

    pref_mean = np.mean(pref_list, axis=0) if pref_list else np.array([0.5, 0.5], dtype=float)
    return pref_mean, rows



def exp_nc_nonlocal(dyn,p,steps,ntraj,A,B, prog: SimpleProgress|None):
    rows=[]; psi0=init_state_zero(p.N); psi1=apply_1q_gate(psi0,X,p.N,A)
    for t in range(steps+1):
        ps0=[]; ps1=[]
        base=t*ntraj
        for k in range(ntraj):
            if prog: prog.update(base+k+1, extra=f"t={t}/{steps}")
            rng=np.random.default_rng(p.seed+50000+97*t+k)
            rng2=np.random.default_rng(); rng2.bit_generator.state=rng.bit_generator.state.copy()
            a=nonlocal_filter_like(psi0.copy(),p,rng,A,B)
            b=nonlocal_filter_like(psi1.copy(),p,rng2,A,B)
            for s in range(t):
                a=dyn.step(a,rng,s,False); b=dyn.step(b,rng2,s,False)
            ps0.append(z_probs_site(a,p.N,B)); ps1.append(z_probs_site(b,p.N,B))
        p0m=np.mean(ps0,axis=0); p1m=np.mean(ps1,axis=0)
        rows.append({"t":t,"tvd":float(tvd(p0m,p1m)),"delta_fail":0.0,"ntraj":ntraj})
    return rows

def exp_nc_postselection(dyn, p, steps, ntraj, A, B, prog: Optional[SimpleProgress]):
    """
    NC4 (post-selection) – garantált conditional eltérés Bob Z-mérésében.

    Konstrukció (kanonikus):
      1) Választunk egy *szomszédos* (u,v) élt a gráfból (NC4_A=u, NC4_B=v).
      2) Bell-pár előállítása: |00> --H_u--> (|0>+|1>)/sqrt2 ⊗ |0> --CNOT_{u->v}--> |Φ+>.
      3) A = u Z-mérése:
         - Unconditional: Bob Z eloszlása maximálisan kevert (≈ [0.5, 0.5]) függetlenül attól, hogy mérünk-e A-n.
         - Conditional: Bob Z eloszlása A kimenetére kondicionálva élesen szétválik (≈ [1,0] vs [0,1]),
           ezért a conditional TVD ≈ 1.

    Megjegyzés:
      - Ez a negatív kontroll *koncepcionális* és topológiától független, ezért a fődinamikától (dyn.step)
        itt nem függünk. Cél: a conditional vs unconditional különbség operacionális demonstrációja.
    """
    # Pick an adjacent edge for guaranteed correlation
    if not hasattr(dyn, "edges") or not dyn.edges:
        # fallback: use provided A,B (may not be adjacent, then guarantee is weaker)
        u, v = A, B
    else:
        # deterministic choice for reproducibility: lexicographically smallest edge
        e = min(dyn.edges, key=lambda e: (min(e[0], e[1]), max(e[0], e[1])))
        u, v = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])

    psi_bell = init_state_zero(p.N)
    psi_bell = apply_1q_gate(psi_bell, H, p.N, u)
    psi_bell = apply_2q_gate(psi_bell, CNOT, p.N, u, v)

    # Unconditional: identity vs. Z-dephasing on A (discard outcome)
    p_un_id = z_probs_site(psi_bell, p.N, v)

    # Unconditional under measurement channel via trajectory average (dephasing = project+discard)
    acc_un = []
    acc0 = []
    acc1 = []
    c0 = 0
    c1 = 0

    # Precompute A outcome probabilities from Bell state (should be 0.5/0.5)
    pA = z_probs_site(psi_bell, p.N, u)  # [p0,p1]
    pA0, pA1 = float(pA[0]), float(pA[1])

    rng = np.random.default_rng(p.seed + 77001)
    for k in range(ntraj):
        if prog:
            prog.update(k + 1, extra="NC4 sampling")
        rr = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        # sample A outcome in Z
        if rr.random() < pA1:
            # outcome 1: collapse to |11> on (u,v) for Bell
            c1 += 1
            # after collapse, Bob is |1>
            acc1.append(np.array([0.0, 1.0], dtype=float))
            acc_un.append(np.array([0.0, 1.0], dtype=float))
        else:
            c0 += 1
            acc0.append(np.array([1.0, 0.0], dtype=float))
            acc_un.append(np.array([1.0, 0.0], dtype=float))

    p_un_meas = np.mean(acc_un, axis=0) if acc_un else np.array([float("nan"), float("nan")])
    tvd_uncond = float(tvd(p_un_id, p_un_meas))  # should be ~0

    pB_A0 = np.mean(acc0, axis=0) if acc0 else np.array([float("nan"), float("nan")])
    pB_A1 = np.mean(acc1, axis=0) if acc1 else np.array([float("nan"), float("nan")])
    tvd_cond = float(tvd(pB_A0, pB_A1)) if (acc0 and acc1) else float("nan")

    return {
        "nc4_A": int(u),
        "nc4_B": int(v),
        "p_un_id_0": float(p_un_id[0]),
        "p_un_id_1": float(p_un_id[1]),
        "p_un_meas_0": float(p_un_meas[0]),
        "p_un_meas_1": float(p_un_meas[1]),
        "tvd_uncond": tvd_uncond,
        "pA0": c0 / float(max(ntraj, 1)),
        "pA1": c1 / float(max(ntraj, 1)),
        "pB_A0_0": float(pB_A0[0]),
        "pB_A0_1": float(pB_A0[1]),
        "pB_A1_0": float(pB_A1[0]),
        "pB_A1_1": float(pB_A1[1]),
        "tvd_cond": tvd_cond,
        "delta_fail": 0.0,
        "ntraj": int(ntraj),
        "notes": "Guaranteed Bell-pair NC4: conditional TVD ~ 1; unconditional TVD ~ 0."
    }


def fig_nosignal(outpath, rows, dist_AB: int):
    """
    No-signaling ábra: TVD(t) Bob Z-mérésén.
    dist_AB csak tájékoztató (graph distance).
    """
    import numpy as np
    t = np.array([r["t"] for r in rows], dtype=int)
    tv = np.array([r["tvd"] for r in rows], dtype=float)
    plt.figure(figsize=(7,5))
    plt.plot(t, tv, "o-", label=r"$D_{\mathrm{TV}}(p_B^0,p_B^1)$ (Z-mérés)")
    if dist_AB is not None and dist_AB >= 0:
        plt.axvline(dist_AB, linestyle="--", color="black", label=f"graph distance d={dist_AB}")
        plt.axvspan(0, dist_AB, alpha=0.12, label="outside the cone (ideally ~0)")
    plt.xlabel(r"number of steps $t$")
    plt.ylabel("TVD")
    plt.title("No-signaling intervention test")
    plt.legend()
    savefig(outpath)

def fig_cone(outpath, infl):
    plt.figure(figsize=(6.6,5))
    plt.imshow(infl,origin="lower",aspect="auto",cmap="inferno",interpolation="nearest")
    plt.colorbar(label=r"$\mathbb{E}[|\Delta\langle Z\rangle|]$")
    plt.xlabel("graph distance $d$"); plt.ylabel("number of steps $t$"); plt.title("Strict cone (distance-band average)")
    savefig(outpath)

def fig_sched(outpath, rows, title):
    vals=np.array([r["tvd_to_ref"] for r in rows],dtype=float)
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=12,alpha=0.75)
    plt.xlabel("TVD relative to the reference schedule"); plt.ylabel("frequency"); plt.title(title)
    savefig(outpath)

def fig_nc_tvd(outpath, rows, title):
    t=np.array([r["t"] for r in rows]); tv=np.array([r["tvd"] for r in rows])
    plt.figure(figsize=(7,5))
    plt.plot(t,tv,"o-"); plt.xlabel(r"number of steps $t$"); plt.ylabel("TVD"); plt.title(title)
    savefig(outpath)

def fig_nc4(outpath, rec):
    plt.figure(figsize=(7,4))
    plt.plot([0,1],[rec["pB_A0_0"],rec["pB_A0_1"]],"o-",label=f"Bob|A=0 (p={rec['pA0']:.2f})")
    plt.plot([0,1],[rec["pB_A1_0"],rec["pB_A1_1"]],"o-",label=f"Bob|A=1 (p={rec['pA1']:.2f})")
    plt.xticks([0,1],["0","1"]); plt.ylabel("probability")
    plt.title(f"NC4: post-selection (uncond={rec['tvd_uncond']:.3g}, cond={rec['tvd_cond']:.3g})")
    plt.legend(); savefig(outpath)

# ---------- Run one topology ----------
def run_one(topology, args, publish, run_prefix):
    # Stable per-topology seed offset (avoid Python hash randomization)
    topo_offset = {"chain": 1001, "star": 2002, "er": 3003}.get(topology, 4004)
    rng = np.random.default_rng(args.seed + topo_offset)

    p=Params(N=args.N,gamma=args.gamma,seed=args.seed,topology=topology,p=args.p)
    if topology=="chain": edges=build_chain(p.N)
    elif topology=="star": edges=build_star(p.N)
    else: edges=build_er_connected(p.N,p.p,rng)

    matchings=greedy_matchings(edges,rng)
    # Edge gate selection (undirected graphs): prefer symmetric mixing gate
    if args.edge_gate == "cnot":
        edge_gate = CNOT
    elif args.edge_gate == "swap":
        edge_gate = SWAP
    else:
        edge_gate = sqrtSWAP
    dyn=GraphDynamics(p,edges,matchings, edge_gate)

    A,B,_=approx_diameter_endpoints(p.N,edges,rng)
    distA=bfs_dist(p.N,edges,A)
    # Choose B to be as far as possible from A, but within steps to avoid a trivial all-zero TVD curve
    candidates = [i for i,d in enumerate(distA) if d >= 0 and d <= args.steps]
    if candidates:
        B = max(candidates, key=lambda i: distA[i])
    dist_AB=distA[B]

    run_id=f"{run_prefix}_{topology}"
    run_root=Path(args.runsdir)/run_id
    logs_dir=run_root/"logs"
    out_dir=run_root/"outputs"
    figs_dir=run_root/"figs"
    ensure_dir(figs_dir)

    # experiments with progress
    prog = SimpleProgress((args.steps+1)*args.ntraj, prefix=f"[{topology}] nosignal", enabled=not args.no_progress, min_interval=args.progress_interval)
    nos=exp_nosignal(dyn,p,args.steps,args.ntraj,A,B, prog)

    prog = SimpleProgress((args.steps+1)*max(200,args.ntraj//6), prefix=f"[{topology}] cone", enabled=not args.no_progress, min_interval=args.progress_interval)
    infl=exp_cone(dyn,p,args.steps,max(200,args.ntraj//6),A,distA, prog)

    prog = SimpleProgress(args.trials*max(200,args.ntraj//6), prefix=f"[{topology}] sched(confluent)", enabled=not args.no_progress, min_interval=args.progress_interval)

    # Scheduling/NC3 paraméterek (alapértelmezés)
    steps_sched = min(6, args.steps)
    B_sched = B
    autotune_info = None
    if args.autotune_nc3 and publish:
        steps_list = [s for s in _parse_int_list(args.autotune_steps) if 1 <= s <= args.steps]
        if not steps_list:
            steps_list = [steps_sched]
        B_candidates = list(range(p.N))
        print(f"[autotune] start: steps={steps_list}, trials={min(args.autotune_trials, args.trials)}, ntraj={min(args.autotune_ntraj, args.ntraj)}, tau={args.autotune_tau}")
        autotune_info = autotune_nc3(
            dyn, p, args.steps, B_candidates, steps_list,
            trials=min(args.autotune_trials, args.trials),
            ntraj=min(args.autotune_ntraj, args.ntraj),
            eps_sched=args.eps_sched,
            tau=args.autotune_tau,
        )
        if autotune_info is not None:
            B_sched = int(autotune_info["B_sched"])
            steps_sched = int(autotune_info["steps_sched"])
    # Felülírások (ha megadod, ezek felülírják a default/autotune választást)
    if args.sched_steps is not None and args.sched_steps > 0:
        steps_sched = min(int(args.sched_steps), int(args.steps))
    if args.sched_B is not None and args.sched_B >= 0:
        B_sched = int(args.sched_B)


    _, sched_good=exp_sched(dyn,p,steps_sched,args.trials,max(200,args.ntraj//6),B_sched,"confluent", prog)

    prog = SimpleProgress(args.trials*max(200,args.ntraj//6), prefix=f"[{topology}] sched(nonconfluent)", enabled=not args.no_progress, min_interval=args.progress_interval)
    _, sched_bad =exp_sched(dyn,p,steps_sched,args.trials,max(200,args.ntraj//6),B_sched,"nonconfluent", prog)

    prog = SimpleProgress((6+1)*max(300,args.ntraj//4), prefix=f"[{topology}] nc1", enabled=not args.no_progress, min_interval=args.progress_interval)
    nc1=exp_nc_nonlocal(dyn,p,6,max(300,args.ntraj//4),A,B, prog)

    prog = SimpleProgress((6+1)*max(300,args.ntraj//4), prefix=f"[{topology}] nc2", enabled=not args.no_progress, min_interval=args.progress_interval)
    nc2=exp_nc_nonlocal(dyn,p,6,max(300,args.ntraj//4),A,B, prog)

    prog = SimpleProgress(max(400,args.ntraj//3), prefix=f"[{topology}] nc4", enabled=not args.no_progress, min_interval=args.progress_interval)
    nc4=exp_nc_postselection(dyn,p,min(6,args.steps),max(400,args.ntraj//3),A,B, prog)

    # outputs
    write_json(run_root/"config.json",{
        "run_id":run_id,
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "model":"CNS reference instantiation: qubit graph + matching layers + damping (trajectories)",
        "params":asdict(p),
        "graph":{"edges":edges,"num_edges":len(edges),"num_matchings":len(matchings)},
        "regions":{"A":A,"B":B,"graph_distance_AB":dist_AB},
        "scheduling_params": {"B_sched": B_sched, "steps_sched": steps_sched, "autotune": bool(args.autotune_nc3 and publish), "sched_override": {"sched_steps": args.sched_steps, "sched_B": args.sched_B}},
        "steps":args.steps,"trials":args.trials,"ntraj":args.ntraj,
        "notes":"Reference instantiation; not a full COS-QD topological simulation."
    })
    write_csv(out_dir/"nosignal.csv",nos,["t","p0_0","p0_1","p1_0","p1_1","tvd","delta_fail","ntraj"])
    ensure_dir(out_dir); np.savez(out_dir/"cone.npz", infl=infl, distA=np.array(distA,dtype=int), A=A, B=B, edges=np.array(edges,dtype=int))
    write_csv(out_dir/"sched_confluent.csv",sched_good,["sample","p0","p1","tvd_to_ref","delta_fail_to_ref","ntraj"])
    write_csv(out_dir/"sched_nonconfluent.csv",sched_bad,["sample","p0","p1","tvd_to_ref","delta_fail_to_ref","ntraj"])
    write_csv(out_dir/"nc1.csv",nc1,["t","tvd","delta_fail","ntraj"])
    write_csv(out_dir/"nc2.csv",nc2,["t","tvd","delta_fail","ntraj"])
    write_csv(out_dir/"nc3.csv",sched_bad,["sample","p0","p1","tvd_to_ref","delta_fail_to_ref","ntraj"])
    write_json(out_dir/"nc4.json",nc4)

    # COS-STAB timeseries
    ensure_dir(logs_dir)
    ts_path=logs_dir/"timeseries.csv"
    with ts_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["step","layer_idx","num_edges_layer","p0_B","p1_B"])
        w.writeheader()
        rr=np.random.default_rng(args.seed+123456)
        psi=init_state_zero(p.N)
        for t in range(args.steps+1):
            pp=z_probs_site(psi,p.N,B); layer=matchings[t%len(matchings)]
            w.writerow({"step":t,"layer_idx":int(t%len(matchings)),"num_edges_layer":len(layer),
                        "p0_B":float(pp[0]),"p1_B":float(pp[1])})
            if t<args.steps: psi=dyn.step(psi,rr,t,False)

    # figures (run-local)
    fig_nosignal(figs_dir/"nosignal-tvd-vs-t.pdf",nos,dist_AB)
    fig_cone(figs_dir/"cone-heatmap.pdf",infl)
    fig_sched(figs_dir/"scheduling-variance.pdf",sched_good,"Scheduling robustness (diszjunkt élek permutációja)")
    fig_nc_tvd(figs_dir/"fig-nc-nonlocal-raw.pdf",nc1,"NC1: non-local nyers lépés (proxy)")
    fig_nc_tvd(figs_dir/"fig-nc-nonlocal-filter.pdf",nc2,"NC2: non-local szűrés (proxy)")
    fig_sched(figs_dir/"fig-nc-scheduling.pdf",sched_bad,"NC3: schedule dependence (non-confluent updates)")
    fig_nc4(figs_dir/"fig-nc-postselection.pdf",nc4)

    # pass/fail
    eps_ns=args.eps_ns; eps_sched=args.eps_sched
    max_tvd_outside = 0.0 if dist_AB<=0 else float(max([r["tvd"] for r in nos[:dist_AB]]))
    max_sched_good = float(np.max([r["tvd_to_ref"] for r in sched_good])) if sched_good else 0.0
    max_sched_bad  = float(np.max([r["tvd_to_ref"] for r in sched_bad])) if sched_bad else 0.0
    summary={
        "run_id":run_id,"topology":topology,
        "thresholds":{"eps_ns":eps_ns,"eps_sched":eps_sched},
        "metrics":{
            "nosignal":{"graph_distance_AB":dist_AB,"max_tvd_outside_cone":max_tvd_outside,"pass": bool(max_tvd_outside<=eps_ns)},
            "sched_confluent":{"max_tvd_to_ref":max_sched_good,"pass": bool(max_sched_good<=eps_sched)},
            "sched_nonconfluent":{"max_tvd_to_ref":max_sched_bad,"expected_violation": bool(max_sched_bad > max(5*eps_sched, 0.02))},
            "nc1":{"max_tvd": float(max([r["tvd"] for r in nc1])) if nc1 else 0.0, "expected_violation": True},
            "nc2":{"max_tvd": float(max([r["tvd"] for r in nc2])) if nc2 else 0.0, "expected_violation": True},
            "nc4":{"tvd_uncond": float(nc4["tvd_uncond"]), "tvd_cond": float(nc4["tvd_cond"])},
        },
        "overall_pass": bool((max_tvd_outside<=eps_ns) and (max_sched_good<=eps_sched)),
        "notes":["General-graph reference instantiation; strict cone speed=1 per matching-layer step."]
    }
    write_json(out_dir/"summary.json", summary)

    if publish:
        outfigs=Path(args.outfigs); ensure_dir(outfigs)
        for name in ["nosignal-tvd-vs-t.pdf","cone-heatmap.pdf","scheduling-variance.pdf"]:
            (outfigs/name).write_bytes((figs_dir/name).read_bytes())
        (outfigs/"fig-nc-nonlocal-raw.pdf").write_bytes((figs_dir/"fig-nc-nonlocal-raw.pdf").read_bytes())
        (outfigs/"fig-nc-nonlocal-filter.pdf").write_bytes((figs_dir/"fig-nc-nonlocal-filter.pdf").read_bytes())
        (outfigs/"fig-nc-scheduling.pdf").write_bytes((figs_dir/"fig-nc-scheduling.pdf").read_bytes())
        (outfigs/"fig-nc-postselection.pdf").write_bytes((figs_dir/"fig-nc-postselection.pdf").read_bytes())

    return run_id, dist_AB, len(edges), len(matchings), summary["metrics"]["nosignal"]["pass"], summary["metrics"]["sched_confluent"]["pass"], summary["overall_pass"]


def _parse_int_list(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out

def autotune_nc3(dyn, p, max_steps: int, B_candidates, steps_candidates, trials: int, ntraj: int, eps_sched: float, tau: float):
    """
    Gyors paraméter-keresés NC3/scheduling informatívvá tételéhez.

    Cél:
      - confluent: max_tvd_good <= eps_sched
      - nonconfluent: max_tvd_bad >= tau

    Score: (max_good - eps)+ + (tau - max_bad)+
    Tie-break: kisebb steps.

    A keresést steps szerint növekvően végezzük, és ha score==0, azonnal megállunk.
    """
    steps_candidates = sorted(set(int(s) for s in steps_candidates if 1 <= int(s) <= max_steps))
    B_candidates = [int(b) for b in B_candidates]

    total = max(1, len(steps_candidates) * len(B_candidates))
    prog = SimpleProgress(total, prefix="[autotune] candidates", enabled=True, min_interval=0.5)
    done = 0

    best = None
    for steps_sched in steps_candidates:
        for B in B_candidates:
            done += 1
            prog.update(done, extra=f"steps={steps_sched}, B={B}")

            # confluent (good)
            _, rows_good = exp_sched(dyn, p, steps_sched, trials, ntraj, B, "confluent", prog=None)
            max_good = float(max([r["tvd_to_ref"] for r in rows_good])) if rows_good else 0.0

            # nonconfluent (bad)
            _, rows_bad = exp_sched(dyn, p, steps_sched, trials, ntraj, B, "nonconfluent", prog=None)
            max_bad = float(max([r["tvd_to_ref"] for r in rows_bad])) if rows_bad else 0.0

            score = max(0.0, max_good - eps_sched) + max(0.0, tau - max_bad)
            cand = {"B_sched": int(B), "steps_sched": int(steps_sched), "max_good": max_good, "max_bad": max_bad, "score": score}

            if best is None or (cand["score"] < best["score"]) or (cand["score"] == best["score"] and cand["steps_sched"] < best["steps_sched"]):
                best = cand

            if score == 0.0:
                return cand

    return best



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runsdir",default="runs")
    ap.add_argument("--outfigs",default="figs")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--N",type=int,default=12)
    ap.add_argument("--gamma",type=float,default=0.05)
    ap.add_argument("--edge_gate", choices=["cnot","swap","sqrt_swap"], default="sqrt_swap", help="Two-qubit gate applied on edges (recommended for undirected graphs: sqrt_swap)")
    ap.add_argument("--steps",type=int,default=10)
    ap.add_argument("--trials",type=int,default=60)
    ap.add_argument("--ntraj",type=int,default=1200)
    ap.add_argument("--topology",choices=["chain","star","er"],default="chain")
    ap.add_argument("--topology_list",default="",help="e.g. chain,star,er")
    ap.add_argument("--publish_topology",default="",help="default: the first one")
    ap.add_argument("--p",type=float,default=0.4)
    ap.add_argument("--eps_ns",type=float,default=1e-3)
    ap.add_argument("--eps_sched",type=float,default=1e-3)
    ap.add_argument("--autotune_nc3", action="store_true", help="NC3/scheduling paraméterek keresése (B és steps) a publish topológiára")
    ap.add_argument("--autotune_steps", default="2,3,4,5,6,8,10,12,16,20", help="steps jelöltek (vesszővel) NC3 kereséshez")
    ap.add_argument("--autotune_trials", type=int, default=20, help="autotune trials (gyors keresés)")
    ap.add_argument("--autotune_ntraj", type=int, default=300, help="autotune ntraj (gyors keresés)")
    ap.add_argument("--autotune_tau", type=float, default=0.05, help="minimum expected non-confluent deviation (TVD) for NC3")
    ap.add_argument("--sched_steps", type=int, default=-1, help="Fix scheduling number of steps (NC3) az autotune felülírására; -1: default/autotune")
    ap.add_argument("--sched_B", type=int, default=-1, help="Fix B csúcs scheduling/NC3 méréshez az autotune felülírására; -1: default/autotune")
    ap.add_argument("--no_progress", action="store_true", help="Disable progress display")
    ap.add_argument("--progress_interval", type=float, default=0.2, help="Minimum progress refresh interval (s)")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)
    run_prefix=now_run_prefix(rng)

    topo_list=[t.strip() for t in (args.topology_list.split(",") if args.topology_list else [args.topology]) if t.strip()]
    if not topo_list: topo_list=[args.topology]
    publish_topo=args.publish_topology.strip() or topo_list[0]

    print(f"Topologies: {topo_list} | publish_topology={publish_topo}")
    rows = []
    for topo in topo_list:
        publish=(topo==publish_topo)
        run_id, dist_ab, m, L, pass_ns, pass_sched, overall = run_one(topo,args,publish,run_prefix)
        rows.append({
            "topology": topo,
            "run_id": run_id,
            "dist_ab": int(dist_ab),
            "num_edges": int(m),
            "num_matchings": int(L),
            "pass_ns": bool(pass_ns),
            "pass_sched": bool(pass_sched),
            "overall": bool(overall),
        })
        print(f"  {topo}: run_id={run_id} |E|={m} matchings={L} dist(A,B)={dist_ab} publish={publish} pass(ns)={pass_ns} pass(sched)={pass_sched} overall={overall}")

    # Print compact table
    print_run_table(rows)

    # Write aggregated index.json under runs/<prefix>/
    prefix_dir = Path(args.runsdir) / run_prefix
    write_index_json(prefix_dir, rows, vars(args))

if __name__=="__main__":
    main()
