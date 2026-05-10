#!/usr/bin/env python3
"""
Build a recurrence / stability map from a completed split-sample Pantheon JSON.

Motivation
----------
A single best split can be unstable. This script instead looks for *clusters*
of repeatedly appearing directions across the split results, using sign-invariant
angular distances on the sphere (v ~ -v). It then exports either the weighted
cluster centroid or cluster medoid as a candidate axis for downstream CMB
fixed-axis follow-up.

Expected input
--------------
A JSON produced by cos_split_sample_crossprobe_standalone_checkpointed.py
(or a compatible variant) containing:
  pantheon_split_validation:
    axis_support_rows: [...]
    split_results: [...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def log(msg: str) -> None:
    print(f"[info] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)


class AnalysisError(RuntimeError):
    pass


@dataclass
class AxisRow:
    split_id: int
    axis_id: Optional[int]
    kind: Optional[str]
    lon_deg: float
    lat_deg: float
    validation_percentile_vs_random: Optional[float]
    validation_fixed_axis_null_p: Optional[float]
    validation_stat: float
    train_stat: Optional[float] = None
    sign_consistent_train_validation: Optional[bool] = None
    aggregate_score: float = 0.0


def unit_vector_from_lonlat_deg(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))
    clat = np.cos(lat)
    return np.array([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)], dtype=float)


def lonlat_from_unit_vector(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = map(float, v)
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


def angular_distance_mod_sign_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    x = float(np.clip(abs(np.dot(v1, v2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(x)))


def weighted_score(
    percentile: Optional[float],
    pval: Optional[float],
    stat: float,
    *,
    w_pct: float,
    w_logp: float,
    w_stat: float,
    sign_bonus: float = 0.0,
    sign_consistent: Optional[bool] = None,
) -> float:
    pct = 0.0 if percentile is None or not np.isfinite(percentile) else float(percentile)
    if pval is None or not np.isfinite(pval) or pval <= 0.0:
        p_term = 0.0
    else:
        p_term = max(0.0, -math.log10(float(pval)))
    stat_term = 0.0 if not np.isfinite(stat) else float(stat)
    sign_term = float(sign_bonus) if sign_consistent is True else 0.0
    return float(w_pct * pct + w_logp * p_term + w_stat * stat_term + sign_term)


def load_rows(path: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], List[AxisRow]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    psv = data.get("pantheon_split_validation")
    if not isinstance(psv, dict):
        raise AnalysisError("Input JSON has no pantheon_split_validation section.")

    raw_rows = psv.get("axis_support_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise AnalysisError("Input JSON has no usable axis_support_rows list.")

    split_map = {int(s["split_id"]): s for s in psv.get("split_results", []) if "split_id" in s}

    rows: List[AxisRow] = []
    for r in raw_rows:
        split_id = int(r["split_id"])
        split_extra = split_map.get(split_id, {})
        val_stat = r.get("validation_stat")
        if val_stat is None:
            # compatibility fallback
            val_stat = split_extra.get("validation_same_axis_stat_value")
        row = AxisRow(
            split_id=split_id,
            axis_id=int(r["axis_id"]) if r.get("axis_id") is not None else None,
            kind=r.get("kind"),
            lon_deg=float(r["lon_deg"]),
            lat_deg=float(r["lat_deg"]),
            validation_percentile_vs_random=(
                None if r.get("validation_percentile_vs_random") is None else float(r["validation_percentile_vs_random"])
            ),
            validation_fixed_axis_null_p=(
                None if r.get("validation_fixed_axis_null_p") is None else float(r["validation_fixed_axis_null_p"])
            ),
            validation_stat=float(val_stat),
            train_stat=(None if r.get("train_stat") is None else float(r["train_stat"])),
            sign_consistent_train_validation=(
                split_extra.get("sign_consistent_train_validation")
                if "sign_consistent_train_validation" in split_extra
                else r.get("sign_consistent_train_validation")
            ),
        )
        row.aggregate_score = weighted_score(
            row.validation_percentile_vs_random,
            row.validation_fixed_axis_null_p,
            row.validation_stat,
            w_pct=args.weight_percentile,
            w_logp=args.weight_logp,
            w_stat=args.weight_stat,
            sign_bonus=args.sign_bonus,
            sign_consistent=row.sign_consistent_train_validation,
        )
        rows.append(row)

    return data, rows


def filter_rows(rows: Sequence[AxisRow], args: argparse.Namespace) -> List[AxisRow]:
    kept: List[AxisRow] = []
    for r in rows:
        if args.require_sign_consistency and r.sign_consistent_train_validation is not True:
            continue
        if args.min_percentile is not None:
            p = r.validation_percentile_vs_random
            if p is None or not np.isfinite(p) or p < args.min_percentile:
                continue
        if args.max_pvalue is not None:
            p = r.validation_fixed_axis_null_p
            if p is None or not np.isfinite(p) or p > args.max_pvalue:
                continue
        if args.min_validation_stat is not None and (not np.isfinite(r.validation_stat) or r.validation_stat < args.min_validation_stat):
            continue
        if r.aggregate_score < args.min_score:
            continue
        kept.append(r)
    return kept


def build_clusters(rows: Sequence[AxisRow], radius_deg: float) -> List[List[int]]:
    if not rows:
        return []
    vecs = [unit_vector_from_lonlat_deg(r.lon_deg, r.lat_deg) for r in rows]
    n = len(rows)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if angular_distance_mod_sign_deg(vecs[i], vecs[j]) <= radius_deg:
                adj[i].add(j)
                adj[j].add(i)

    clusters: List[List[int]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        clusters.append(sorted(comp))
    return clusters


def cluster_centroid(rows: Sequence[AxisRow], indices: Sequence[int]) -> Tuple[np.ndarray, float]:
    if not indices:
        raise AnalysisError("Empty cluster.")
    vecs = [unit_vector_from_lonlat_deg(rows[i].lon_deg, rows[i].lat_deg) for i in indices]
    weights = np.array([max(rows[i].aggregate_score, 0.0) for i in indices], dtype=float)

    ref = vecs[int(np.argmax(weights)) if np.any(weights > 0.0) else 0]
    vec_sum = np.zeros(3, dtype=float)
    for v, w in zip(vecs, weights):
        vv = v.copy()
        if np.dot(vv, ref) < 0.0:
            vv = -vv
        vec_sum += (w if w > 0 else 1.0) * vv
    if np.linalg.norm(vec_sum) == 0.0:
        vec_sum = ref.copy()
    centroid = vec_sum / np.linalg.norm(vec_sum)

    dists = []
    for v in vecs:
        vv = v.copy()
        if np.dot(vv, centroid) < 0.0:
            vv = -vv
        dists.append(angular_distance_mod_sign_deg(centroid, vv))
    return centroid, float(np.mean(dists))


def cluster_medoid_index(rows: Sequence[AxisRow], indices: Sequence[int], centroid_vec: np.ndarray) -> int:
    best_i = indices[0]
    best_d = float("inf")
    for i in indices:
        v = unit_vector_from_lonlat_deg(rows[i].lon_deg, rows[i].lat_deg)
        d = angular_distance_mod_sign_deg(v, centroid_vec)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def summarise_cluster(cluster_id: int, rows: Sequence[AxisRow], indices: Sequence[int]) -> Dict[str, Any]:
    centroid_vec, mean_dist = cluster_centroid(rows, indices)
    cen_lon, cen_lat = lonlat_from_unit_vector(centroid_vec)
    medoid_i = cluster_medoid_index(rows, indices, centroid_vec)

    vals_pct = np.array([rows[i].validation_percentile_vs_random for i in indices if rows[i].validation_percentile_vs_random is not None and np.isfinite(rows[i].validation_percentile_vs_random)], dtype=float)
    vals_p = np.array([rows[i].validation_fixed_axis_null_p for i in indices if rows[i].validation_fixed_axis_null_p is not None and np.isfinite(rows[i].validation_fixed_axis_null_p)], dtype=float)
    vals_stat = np.array([rows[i].validation_stat for i in indices if np.isfinite(rows[i].validation_stat)], dtype=float)
    scores = np.array([rows[i].aggregate_score for i in indices], dtype=float)

    strong_members = int(np.sum(vals_p < 0.10)) if vals_p.size > 0 else 0
    very_strong_members = int(np.sum(vals_p < 0.05)) if vals_p.size > 0 else 0
    unique_axis_ids = sorted({int(rows[i].axis_id) for i in indices if rows[i].axis_id is not None})

    members = []
    for i in sorted(indices, key=lambda j: rows[j].aggregate_score, reverse=True):
        r = rows[i]
        members.append(
            {
                "split_id": r.split_id,
                "axis_id": r.axis_id,
                "kind": r.kind,
                "lon_deg": r.lon_deg,
                "lat_deg": r.lat_deg,
                "validation_percentile_vs_random": r.validation_percentile_vs_random,
                "validation_fixed_axis_null_p": r.validation_fixed_axis_null_p,
                "validation_stat": r.validation_stat,
                "aggregate_score": r.aggregate_score,
                "sign_consistent_train_validation": r.sign_consistent_train_validation,
            }
        )

    medoid = rows[medoid_i]
    out = {
        "cluster_id": int(cluster_id),
        "member_count": int(len(indices)),
        "unique_axis_id_count": int(len(unique_axis_ids)),
        "unique_axis_ids": unique_axis_ids,
        "weight_sum": float(np.sum(scores)),
        "weight_mean": float(np.mean(scores)),
        "median_validation_percentile_vs_random": float(np.median(vals_pct)) if vals_pct.size > 0 else None,
        "mean_validation_percentile_vs_random": float(np.mean(vals_pct)) if vals_pct.size > 0 else None,
        "median_validation_fixed_axis_null_p": float(np.median(vals_p)) if vals_p.size > 0 else None,
        "mean_validation_fixed_axis_null_p": float(np.mean(vals_p)) if vals_p.size > 0 else None,
        "mean_validation_stat": float(np.mean(vals_stat)) if vals_stat.size > 0 else None,
        "strong_members_p_lt_0p10": strong_members,
        "very_strong_members_p_lt_0p05": very_strong_members,
        "centroid_lon_deg": float(cen_lon),
        "centroid_lat_deg": float(cen_lat),
        "mean_angular_scatter_deg": float(mean_dist),
        "medoid_split_id": int(medoid.split_id),
        "medoid_axis_id": medoid.axis_id,
        "medoid_lon_deg": float(medoid.lon_deg),
        "medoid_lat_deg": float(medoid.lat_deg),
        "members": members,
    }
    return out


def cluster_rank_value(cluster: Dict[str, Any], mode: str) -> float:
    if mode == "weight_sum":
        return float(cluster["weight_sum"])
    if mode == "weight_mean":
        return float(cluster["weight_mean"])
    if mode == "recurrence":
        return float(cluster["member_count"])
    if mode == "hybrid":
        return float(cluster["weight_sum"]) * math.sqrt(float(cluster["member_count"]))
    raise AnalysisError(f"Unknown ranking mode: {mode}")


def choose_export_axis(
    clusters: Sequence[Dict[str, Any]],
    *,
    ranking_mode: str,
    export_kind: str,
    min_cluster_members: int,
) -> Optional[Dict[str, Any]]:
    eligible = [c for c in clusters if int(c["member_count"]) >= int(min_cluster_members)]
    if not eligible:
        return None

    best = max(eligible, key=lambda c: cluster_rank_value(c, ranking_mode))
    if export_kind == "centroid":
        lon_deg = float(best["centroid_lon_deg"])
        lat_deg = float(best["centroid_lat_deg"])
        source = "recurrence_cluster_centroid"
    elif export_kind == "medoid":
        lon_deg = float(best["medoid_lon_deg"])
        lat_deg = float(best["medoid_lat_deg"])
        source = "recurrence_cluster_medoid"
    else:
        raise AnalysisError(f"Unknown export kind: {export_kind}")

    return {
        "source": source,
        "cluster_id": int(best["cluster_id"]),
        "ranking_mode": ranking_mode,
        "export_kind": export_kind,
        "lon_deg": lon_deg,
        "lat_deg": lat_deg,
        "member_count": int(best["member_count"]),
        "weight_sum": float(best["weight_sum"]),
        "weight_mean": float(best["weight_mean"]),
        "strong_members_p_lt_0p10": int(best["strong_members_p_lt_0p10"]),
        "very_strong_members_p_lt_0p05": int(best["very_strong_members_p_lt_0p05"]),
        "mean_angular_scatter_deg": float(best["mean_angular_scatter_deg"]),
    }


def make_mollweide_plot(
    rows: Sequence[AxisRow],
    clusters: Sequence[Dict[str, Any]],
    export_axis: Optional[Dict[str, Any]],
    outpath: str,
) -> None:
    if plt is None:
        warn("matplotlib is not available; skipping plot.")
        return

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, alpha=0.35)

    if rows:
        lons = []
        lats = []
        sizes = []
        labels_done = set()
        member_to_cluster = {}
        for c in clusters:
            for m in c["members"]:
                member_to_cluster[(int(m["split_id"]), int(m["axis_id"]) if m["axis_id"] is not None else None)] = int(c["cluster_id"])

        cmap = plt.get_cmap("tab10")
        for r in rows:
            cluster_id = member_to_cluster.get((int(r.split_id), r.axis_id), -1)
            lon = ((r.lon_deg + 180.0) % 360.0) - 180.0
            lon = -lon  # astronomical convention flip for nicer Mollweide view
            lat = r.lat_deg
            lons.append(np.deg2rad(lon))
            lats.append(np.deg2rad(lat))
            sizes.append(30.0 + 80.0 * max(r.aggregate_score, 0.0))
        colors = []
        for r in rows:
            cluster_id = member_to_cluster.get((int(r.split_id), r.axis_id), -1)
            colors.append("0.7" if cluster_id < 0 else cmap(cluster_id % 10))
        ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.8, edgecolors="black", linewidths=0.5)

    for c in clusters:
        lon = ((float(c["centroid_lon_deg"]) + 180.0) % 360.0) - 180.0
        lon = -lon
        lat = float(c["centroid_lat_deg"])
        ax.scatter(
            [np.deg2rad(lon)],
            [np.deg2rad(lat)],
            s=220,
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            label=f'cluster {c["cluster_id"]} centroid (n={c["member_count"]})',
        )

    if export_axis is not None:
        lon = ((float(export_axis["lon_deg"]) + 180.0) % 360.0) - 180.0
        lon = -lon
        lat = float(export_axis["lat_deg"])
        ax.scatter(
            [np.deg2rad(lon)],
            [np.deg2rad(lat)],
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            label="export axis",
        )

    ax.set_title("Pantheon split-sample axis recurrence map")
    if len(clusters) <= 8:
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    log(f"Saved plot: {outpath}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a recurrence / stability map from split-sample Pantheon axis results."
    )
    ap.add_argument("--split-json", required=True, help="Completed split_crossprobe.json file")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--plot", default=None, help="Optional Mollweide PNG path")

    ap.add_argument("--cluster-radius-deg", type=float, default=25.0, help="Sign-invariant angular linkage radius")
    ap.add_argument("--min-cluster-members", type=int, default=2, help="Minimum members required for export eligibility")
    ap.add_argument("--ranking-mode", choices=["weight_sum", "weight_mean", "recurrence", "hybrid"], default="hybrid")
    ap.add_argument("--export-kind", choices=["centroid", "medoid"], default="centroid")

    ap.add_argument("--weight-percentile", type=float, default=1.0)
    ap.add_argument("--weight-logp", type=float, default=1.0)
    ap.add_argument("--weight-stat", type=float, default=0.05)
    ap.add_argument("--sign-bonus", type=float, default=0.0)

    ap.add_argument("--min-score", type=float, default=0.0, help="Drop rows below this aggregate score before clustering")
    ap.add_argument("--min-percentile", type=float, default=None, help="Optional validation percentile cut")
    ap.add_argument("--max-pvalue", type=float, default=None, help="Optional validation null p-value cut")
    ap.add_argument("--min-validation-stat", type=float, default=None, help="Optional validation statistic cut")
    ap.add_argument("--require-sign-consistency", action="store_true")

    ap.add_argument("--emit-cmb-command", action="store_true", help="Write a ready-to-run command snippet into the JSON")
    ap.add_argument("--crossprobe-script", default="cos_crossprobe_fixed_axis_patched.py")
    ap.add_argument("--map", default="COM_CMB_IQU-smica_2048_R3.00_full.fits")
    ap.add_argument("--mask", default="COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
    ap.add_argument("--work-nside", type=int, default=256)
    ap.add_argument("--lmax-grid", default="8,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--mi-estimator", choices=["hist", "knn"], default="knn")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--mi-sample-size", type=int, default=20000)
    ap.add_argument("--n-phase-null", type=int, default=200)
    ap.add_argument("--run-bayes", action="store_true")
    ap.add_argument("--bayes-nlive", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data, rows_all = load_rows(args.split_json, args)
    rows = filter_rows(rows_all, args)

    log(f"Loaded {len(rows_all)} axis-support rows from: {args.split_json}")
    log(f"Rows kept for clustering after filters: {len(rows)}")
    if not rows:
        raise AnalysisError("No rows survived the requested filters.")

    cluster_indices = build_clusters(rows, args.cluster_radius_deg)
    summaries = [summarise_cluster(i, rows, idxs) for i, idxs in enumerate(cluster_indices)]
    summaries.sort(key=lambda c: cluster_rank_value(c, args.ranking_mode), reverse=True)
    for rank, c in enumerate(summaries, start=1):
        c["rank_by_" + args.ranking_mode] = rank
        c["rank_value"] = cluster_rank_value(c, args.ranking_mode)

    export_axis = choose_export_axis(
        summaries,
        ranking_mode=args.ranking_mode,
        export_kind=args.export_kind,
        min_cluster_members=args.min_cluster_members,
    )
    if export_axis is None:
        warn("No cluster satisfied the minimum member threshold for export.")
    else:
        log(
            "Recommended export axis: "
            f"cluster={export_axis['cluster_id']}, "
            f"(lon,lat)=({export_axis['lon_deg']:.3f},{export_axis['lat_deg']:.3f}), "
            f"members={export_axis['member_count']}, "
            f"weight_sum={export_axis['weight_sum']:.3f}"
        )

    cmb_cmd = None
    if export_axis is not None and args.emit_cmb_command:
        cmd = [
            "python3", args.crossprobe_script,
            "--axis-lon", f"{export_axis['lon_deg']:.10f}",
            "--axis-lat", f"{export_axis['lat_deg']:.10f}",
            "--axis-coords", "gal",
            "--map", args.map,
            "--mask", args.mask,
            "--work-nside", str(args.work_nside),
            "--lmax-grid", args.lmax_grid,
            "--mi-estimator", args.mi_estimator,
            "--knn-k", str(args.knn_k),
            "--mi-sample-size", str(args.mi_sample_size),
            "--n-phase-null", str(args.n_phase_null),
            "--out", "crossprobe_recurrence_axis.json",
            "--plot", "crossprobe_recurrence_axis.png",
        ]
        if args.mi_estimator != "knn":
            # keep command simple; hist users can edit bins manually if needed
            pass
        if args.run_bayes:
            cmd.extend(["--run-bayes", "--bayes-nlive", str(args.bayes_nlive)])
        cmb_cmd = " \\\n  ".join(cmd)

    out = {
        "input_split_json": os.path.abspath(args.split_json),
        "filtering": {
            "cluster_radius_deg": float(args.cluster_radius_deg),
            "min_cluster_members": int(args.min_cluster_members),
            "ranking_mode": args.ranking_mode,
            "export_kind": args.export_kind,
            "min_score": float(args.min_score),
            "min_percentile": args.min_percentile,
            "max_pvalue": args.max_pvalue,
            "min_validation_stat": args.min_validation_stat,
            "require_sign_consistency": bool(args.require_sign_consistency),
            "weights": {
                "percentile": float(args.weight_percentile),
                "logp": float(args.weight_logp),
                "stat": float(args.weight_stat),
                "sign_bonus": float(args.sign_bonus),
            },
        },
        "rows_summary": {
            "n_rows_total": int(len(rows_all)),
            "n_rows_used": int(len(rows)),
        },
        "clusters": summaries,
        "recommended_export_axis": export_axis,
        "cmb_command": cmb_cmd,
        "pantheon_split_metadata": data.get("pantheon_split_validation", {}).get("metadata"),
        "pantheon_previous_best_validated_axis": data.get("pantheon_split_validation", {}).get("best_validated_axis"),
        "pantheon_previous_consensus_axis": data.get("pantheon_split_validation", {}).get("consensus_axis"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    tmp.replace(out_path)
    log(f"Saved output: {args.out}")

    if args.plot:
        make_mollweide_plot(rows, summaries, export_axis, args.plot)


if __name__ == "__main__":
    main()
