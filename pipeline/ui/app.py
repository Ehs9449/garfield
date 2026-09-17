#!/usr/bin/env python3
"""
Drone-to-BIM pipeline control panel.

Start it from the repository root:

    conda activate nerfstudio3
    streamlit run pipeline/ui/app.py

Then open http://localhost:8501 in your browser.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_io as io  # noqa: E402

st.set_page_config(
    page_title="Drone-to-BIM pipeline",
    page_icon="🏢",
    layout="wide",
)

PLOT_BG = "rgba(0,0,0,0)"


def _version_tuple(text):
    parts = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# Streamlit 1.50 replaced use_container_width with width="stretch".
# Both spellings are kept so the page works on either version.
FIT = ({"width": "stretch"}
       if _version_tuple(st.__version__) >= (1, 50, 0)
       else {"use_container_width": True})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def base_layout(fig, height=420, **kwargs):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(size=13),
        **kwargs,
    )
    return fig


def bar_chart(labels, values, title, color=io.SERIES_1, text=None, horizontal=True):
    """One series, so magnitude only: a single hue, sorted, with direct labels."""
    fig = go.Figure()
    if horizontal:
        fig.add_bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=text if text is not None else values,
            textposition="outside",
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
        fig.update_yaxes(autorange="reversed")
    else:
        fig.add_bar(
            x=labels,
            y=values,
            marker=dict(color=color, line=dict(width=0)),
            text=text if text is not None else values,
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return base_layout(fig, title=title, showlegend=False,
                       height=max(260, 26 * len(labels) + 90) if horizontal else 360)


def point_cloud_figure(points, colors, hover, height=680, size=1.6):
    fig = go.Figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=size, color=colors, opacity=0.9),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, title=""),
            yaxis=dict(showbackground=False, title=""),
            zaxis=dict(showbackground=False, title=""),
        )
    )
    return base_layout(fig, height=height, showlegend=False)


def subsample(n_total, n_max, seed=0):
    if n_total <= n_max:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_max, replace=False)


def show_image(target, image, caption=None):
    """st.image changed its width keyword three times across versions.

    1.39 and older take use_column_width, 1.40-1.49 use_container_width,
    1.50+ width="stretch". Try them in order and fall back to no keyword.
    """
    for kwargs in (FIT, {"use_container_width": True},
                   {"use_column_width": True}, {}):
        try:
            target.image(image, caption=caption, **kwargs)
            return
        except TypeError:
            continue


def safe_slider(target, label, low, high, value=None, step=1, help=None):
    """A slider that copes with a range of one.

    Streamlit refuses min_value == max_value, which happens whenever there is
    a single camera, a single grouping level, and so on.
    """
    low, high = int(low), int(high)
    if high <= low:
        target.caption(f"{label}: {low} (only one choice)")
        return low
    if value is None:
        value = low
    value = max(low, min(int(value), high))
    return target.slider(label, low, high, value, step, help=help)


def show_table(rows, **kwargs):
    """A table that still renders when pyarrow cannot be imported.

    st.dataframe, st.data_editor and even st.write all reach into pyarrow for
    anything that looks tabular, so the fallback builds markdown by hand and
    never hands a sequence back to streamlit.
    """
    try:
        rows = list(rows or [])
    except TypeError:
        rows = []

    try:
        st.dataframe(rows, **kwargs, **FIT)
        return
    except Exception as exc:
        st.caption(f"(plain table — {type(exc).__name__})")

    if not rows:
        return

    if isinstance(rows[0], dict):
        columns = []
        for row in rows[:200]:
            for key in row:
                if key not in columns:
                    columns.append(str(key))
        body = [[str(row.get(c, "")) for c in columns] for row in rows[:200]]
    else:
        width = max(len(r) for r in rows[:200])
        columns = [f"col {i + 1}" for i in range(width)]
        body = [[str(v) for v in list(r)] + [""] * (width - len(r))
                for r in rows[:200]]

    def clean(text):
        return text.replace("|", "\\|").replace("\n", " ")[:120]

    lines = ["| " + " | ".join(clean(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(clean(v) for v in row) + " |" for row in body]
    st.markdown("\n".join(lines))

    if len(rows) > 200:
        st.caption(f"showing the first 200 of {len(rows):,} rows")


def show_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

if "cfg" not in st.session_state:
    st.session_state.cfg = io.load_config()

cfg = st.session_state.cfg
paths = io.stage_paths(cfg)

with st.sidebar:
    st.markdown("### Pipeline")
    st.caption(f"repo `{io.REPO_ROOT}`")
    st.write(f"**dataset** `{paths['dataset_name']}`")
    st.write(f"**outputs** `outputs/{paths['dataset_name']}`")

    if st.button("Reload config from disk", **FIT):
        st.session_state.cfg = io.load_config()
        st.rerun()

    st.divider()
    st.markdown("### Stage status")
    for row in io.stage_status(cfg):
        mark = "✅" if row["done"] else "⬜"
        st.write(f"{mark} {row['label']}")
        if row["done"]:
            st.caption(f"　 {row['mtime']} · {row['size_mb']} MB")

    st.divider()
    st.session_state.setdefault("cmd_prefix", "")
    st.session_state.cmd_prefix = st.text_input(
        "Command prefix",
        value=st.session_state.cmd_prefix,
        help="Leave empty unless snakemake is not on PATH. "
             "Example: conda run -n base",
    )

    if not io.comments_preserved():
        st.warning(
            "`ruamel.yaml` is not installed, so saving config.yaml will drop "
            "your comments. A timestamped backup is always written first.\n\n"
            "`pip install ruamel.yaml`"
        )


tab_status, tab_settings, tab_run, tab_proj, tab_cameras, tab_clusters, \
    tab_merge, tab_semantic, tab_masks, tab_history = st.tabs([
        "Overview",
        "Settings",
        "Run",
        "Projection views",
        "Cameras",
        "Clusters",
        "Merge evidence",
        "Semantic result",
        "SAM3 masks",
        "History",
    ])


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------

with tab_status:
    st.subheader("Where the pipeline stands")

    metrics = io.collect_metrics(cfg)
    show_metric_row([
        ("Points", f"{metrics.get('points', 0):,}",
         "Points in the cropped cloud."),
        ("Coverage", f"{metrics.get('coverage_pct', 0)}%",
         "Share of points seen by at least one view. The rest copy a neighbour."),
        ("Views", metrics.get("n_views", "–"),
         "Orthographic views used in the projection."),
        ("Side share", f"{metrics.get('side_share_pct', 0)}%",
         "Share of all feature samples that came from facade views, not roof views."),
    ])
    show_metric_row([
        ("Clusters", metrics.get("n_clusters", "–"),
         "Clusters found by HDBSCAN on the full cloud."),
        ("Noise", f"{metrics.get('noise_pct', 0)}%",
         "Points HDBSCAN left unassigned."),
        ("After merge", metrics.get("clusters_after_merge", "–"),
         "Clusters left once the verified merges were applied."),
        ("Unknown", f"{metrics.get('pct_unknown', 0)}%",
         "Share of labeled points that got no semantic vote."),
    ])

    st.divider()
    rows = io.stage_status(cfg)
    for row in rows:
        c1, c2, c3 = st.columns([3, 5, 2])
        c1.write(("✅ " if row["done"] else "⬜ ") + f"**{row['label']}**")
        c2.caption(row["note"])
        c3.caption(row["mtime"] if row["done"] else "not run yet")


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

def num_input(section, key, label, help_text, *, step=None, fmt=None,
              min_value=None, max_value=None, is_int=False):
    current = cfg.get(section, {}).get(key)
    if current is None:
        st.caption(f"`{section}.{key}` is not in config.yaml")
        return
    if is_int:
        value = st.number_input(label, value=int(current), step=step or 1,
                                min_value=min_value, max_value=max_value,
                                help=help_text, key=f"w_{section}_{key}")
    else:
        value = st.number_input(label, value=float(current),
                                step=step or 0.01, format=fmt or "%.4f",
                                min_value=min_value, max_value=max_value,
                                help=help_text, key=f"w_{section}_{key}")
    cfg.setdefault(section, {})[key] = int(value) if is_int else float(value)


def list_input(section, key, label, help_text):
    current = cfg.get(section, {}).get(key)
    if current is None:
        st.caption(f"`{section}.{key}` is not in config.yaml")
        return
    text = st.text_input(label, value=", ".join(str(v) for v in current),
                         help=help_text, key=f"w_{section}_{key}")
    try:
        cfg.setdefault(section, {})[key] = [
            float(part) for part in text.replace(";", ",").split(",") if part.strip()
        ]
    except ValueError:
        st.error(f"{label}: use numbers separated by commas")


with tab_settings:
    st.subheader("Settings")
    st.caption(
        "These write into `pipeline/config.yaml`. Nothing is saved until you "
        "press the save button at the bottom."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Orthographic projection")
        num_input("projection", "scale", "GARField scale",
                  "Group size GARField is asked for, in nerf units. Small = finer "
                  "parts. A 1.5 m window on this building is about 0.0125.",
                  step=0.005, fmt="%.4f", min_value=0.0001)
        num_input("projection", "n_side", "Side views (azimuth steps)",
                  "4 = the four facades. 8 or 12 adds corner views and raises the "
                  "share of facade features.", is_int=True, min_value=1, max_value=36)
        list_input("projection", "side_elevations", "Side elevations (degrees)",
                   "0 is horizontal. Add a second value such as 20 for a "
                   "second ring of facade views.")
        num_input("projection", "azimuth_offset", "Azimuth offset (degrees)",
                  "Rotate every view so they line up with the real facades.",
                  step=5.0, fmt="%.1f")
        num_input("projection", "n_top_ring", "Roof views", "Steep views around "
                  "the roof. One true top-down view is always added.",
                  is_int=True, min_value=0, max_value=24)
        list_input("projection", "top_elevations", "Roof elevations (degrees)",
                   "80 looks steeply down. 90 would be straight down.")
        num_input("projection", "dist_factor", "Distance factor",
                  "Image plane distance as a multiple of the crop box "
                  "half-diagonal. 1.5 is safe.", step=0.1, fmt="%.2f",
                  min_value=1.0)
        num_input("projection", "resolution", "Resolution (long side, px)",
                  "Pixels on the long side of each view. Raise it to resolve "
                  "windows. Cost grows with the square.",
                  is_int=True, step=270, min_value=256, max_value=8192)

        st.markdown("#### Clustering")
        num_input("clustering", "n_trials", "Optuna trials",
                  "How many HDBSCAN parameter sets to try.",
                  is_int=True, min_value=1, max_value=500)
        num_input("clustering", "sample_size", "Tuning sample size",
                  "Points used during the search. The final run uses all points.",
                  is_int=True, step=10000, min_value=1000)
        num_input("clustering", "optimization_min_clusters", "Min clusters",
                  "Trials below this count are rejected.",
                  is_int=True, min_value=2)
        num_input("clustering", "optimization_max_clusters", "Max clusters",
                  "Trials above this count are rejected. Keep it high to favour "
                  "oversegmentation, which merging can undo.",
                  is_int=True, min_value=2)

    with right:
        st.markdown("#### Labeling view rings")
        st.caption("Cameras sit on rings around the building and look at its centre.")
        rings = cfg.get("labeling_views", {}).get("rings") or []
        current = [{"elevation": float(r["elevation"]),
                    "n_azimuth": int(r["n_azimuth"])} for r in rings]
        try:
            edited = st.data_editor(
                current,
                num_rows="dynamic",
                key="w_rings",
                column_config={
                    "elevation": st.column_config.NumberColumn(
                        "Elevation (deg)", min_value=-30.0, max_value=89.0,
                        step=5.0),
                    "n_azimuth": st.column_config.NumberColumn(
                        "Views on this ring", min_value=1, max_value=64, step=1),
                },
                **FIT,
            )
        except Exception as exc:
            # st.data_editor needs pyarrow; fall back to plain text.
            st.caption(f"(text entry: {type(exc).__name__})")
            text = st.text_input(
                "Rings as elevation@count, separated by commas",
                ", ".join(f"{r['elevation']:g}@{r['n_azimuth']}" for r in current),
                key="w_rings_text",
            )
            edited = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    elev, count = part.split("@")
                    edited.append({"elevation": float(elev),
                                   "n_azimuth": int(count)})
                except ValueError:
                    st.error(f"Could not read '{part}'. Use 20@16 for "
                             "16 views at 20 degrees.")
        cfg.setdefault("labeling_views", {})["rings"] = [
            {"elevation": float(r["elevation"]), "n_azimuth": int(r["n_azimuth"])}
            for r in edited
            if r.get("n_azimuth")
        ]
        total_views = sum(r["n_azimuth"] for r in cfg["labeling_views"]["rings"])
        st.caption(
            f"{total_views} ring views + "
            f"{cfg.get('labeling_views', {}).get('n_top', 1)} top view(s). "
            "Every one of these is sent to the HPC for SAM3."
        )
        num_input("labeling_views", "n_top", "Top-down views",
                  "Straight-down views for the roof.", is_int=True,
                  min_value=0, max_value=8)
        num_input("labeling_views", "radius_factor", "Radius factor",
                  "Margin around the building in the image. 1.15 leaves a small "
                  "border. Raise it if the building is cut off.",
                  step=0.05, fmt="%.2f", min_value=1.0)

        st.markdown("#### Cluster merging")
        num_input("merging", "min_cluster_fraction", "Min cluster fraction",
                  "How much of a cluster must sit inside one SAM2 mask before "
                  "that mask counts as evidence.", step=0.05, fmt="%.2f",
                  min_value=0.0, max_value=1.0)
        num_input("merging", "min_support", "Min supporting views",
                  "A pair needs this many cameras agreeing before it can merge.",
                  is_int=True, min_value=1)
        num_input("merging", "min_cosine", "Min feature cosine",
                  "GARField feature similarity a pair must also reach.",
                  step=0.01, fmt="%.3f", min_value=0.0, max_value=1.0)
        num_input("merging", "max_elevation_deg", "Max camera elevation",
                  "Cameras looking down more steeply than this are not used as "
                  "merge evidence. 0 is a horizontal side view.",
                  step=5.0, fmt="%.1f", min_value=0.0, max_value=90.0)
        num_input("merging", "max_incidence_deg", "Max facade incidence",
                  "How far a camera may be from facing a facade head-on.",
                  step=5.0, fmt="%.1f", min_value=0.0, max_value=90.0)
        num_input("merging", "max_offaxis_deg", "Max off-axis angle",
                  "The camera must point within this angle of the building centre.",
                  step=5.0, fmt="%.1f", min_value=0.0, max_value=90.0)
        list_input("merging", "facade_azimuths", "Facade azimuths (degrees)",
                   "Directions your facades face, measured around +Z from +X. "
                   "Change these if the building is not aligned with the axes.")

        st.markdown("#### Semantic matching")
        num_input("matching", "min_iou", "Min IoU",
                  "A cluster must overlap a SAM3 mask by at least this much "
                  "before the mask may vote on its label.",
                  step=0.01, fmt="%.3f", min_value=0.0, max_value=1.0)

    st.divider()
    save_col, info_col = st.columns([1, 4])
    if save_col.button("Save to config.yaml", type="primary",
                       **FIT):
        backup = io.save_config(cfg)
        st.success(f"Saved. Backup written to `{backup.name}`.")
    info_col.caption(
        "A backup of the previous config is written every time you save. "
        "Snakemake sees config.yaml as an input, so the stages that use a "
        "changed value will re-run on their own."
    )

    with st.expander("Show the current config.yaml as text"):
        st.code(io.CONFIG_PATH.read_text(), language="yaml")


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def run_stage(rule: str, target: Path, dry: bool):
    cmd = io.snakemake_command(rule, target, dry=dry,
                               prefix=st.session_state.cmd_prefix)
    st.code(" ".join(cmd), language="bash")

    log_path = None if dry else io.new_log_path(rule)
    out_box = st.empty()
    tail = deque(maxlen=400)
    started = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=io.REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    handle = open(log_path, "w") if log_path else None
    try:
        for line in process.stdout:
            tail.append(line.rstrip())
            if handle:
                handle.write(line)
                handle.flush()
            out_box.code("\n".join(tail), language="text")
    finally:
        process.wait()
        if handle:
            handle.close()

    seconds = time.time() - started
    if process.returncode == 0:
        st.success(f"Finished in {seconds:.0f} s")
    else:
        st.error(f"Stopped with exit code {process.returncode} after {seconds:.0f} s")

    if not dry:
        io.append_run(io.load_config(), rule, process.returncode, seconds, log_path)
        st.caption(f"Log saved to `{log_path.relative_to(io.REPO_ROOT)}`. "
                   "The run was added to the History tab.")


with tab_run:
    st.subheader("Run a stage")
    st.caption(
        "Every command uses `--rerun-triggers mtime` and `--allowed-rules`, so "
        "only the stage you pick can run. Training is never touched."
    )

    status = {r["key"]: r for r in io.stage_status(cfg)}

    for stage in io.STAGES:
        row = status[stage["key"]]
        c1, c2, c3, c4 = st.columns([4, 1.3, 1.3, 2])
        c1.write(("✅ " if row["done"] else "⬜ ") + f"**{stage['label']}**")
        c1.caption(stage["note"])

        dry = c2.button("Check", key=f"dry_{stage['key']}",
                        help="Dry run. Shows what would happen, changes nothing.")
        go_it = c3.button("Run", key=f"run_{stage['key']}",
                          **FIT, type="primary")
        c4.caption(row["mtime"] if row["done"] else "no output yet")

        if stage["key"] == "sam3_inference":
            c4.caption("needs SSH to the HPC")

        if dry or go_it:
            with st.container():
                run_stage(stage["key"], row["target"], dry=dry)

    st.divider()
    st.markdown("#### Everything that is out of date")
    c1, c2 = st.columns([1, 4])
    if c1.button("Dry run (all)", **FIT):
        run_stage("", paths["semantic"] / "semantic_pointcloud.ply", dry=True)
    c2.caption(
        "This checks the whole workflow, including the HPC step. Use the "
        "per-stage buttons above to actually run things."
    )


# ----------------------------------------------------------------------
# Projection views
# ----------------------------------------------------------------------

with tab_proj:
    st.subheader("Orthographic projection")

    views_dir = paths["proj"] / "views"
    stats, source = io.read_view_stats(paths["proj"])

    if stats:
        bal = io.view_balance(stats)
        st.caption(f"per-view numbers read from {source}")
        show_metric_row([
            ("Views", bal["n_views"], "Views rendered."),
            ("Side samples", f"{bal['side_samples']:,}",
             "Point-view samples taken from facade views."),
            ("Top samples", f"{bal['top_samples']:,}",
             "Point-view samples taken from roof and top-down views."),
            ("Side share", f"{bal['side_share']:.1f}%",
             "Every point averages the features of all views that see it. "
             "A low number here means roof views dominate the facade features."),
        ])

        if bal["side_share"] < 35:
            st.warning(
                f"Only {bal['side_share']:.0f}% of the feature samples come from "
                "facade views. Roof views are diluting the facade features, which "
                "hurts window separation. Raise **Side views** and add a second "
                "**side elevation** in Settings."
            )

        views = stats["views"]
        names = [v["name"] for v in views]
        values = [v["visible_points"] for v in views]
        kinds = ["side" if n.startswith("side") else "top" for n in names]

        fig = go.Figure()
        for kind, color in (("side", io.SERIES_1), ("top", io.SERIES_2)):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            if not idx:
                continue
            fig.add_bar(
                y=[names[i] for i in idx],
                x=[values[i] for i in idx],
                orientation="h",
                name=f"{kind} views",
                marker=dict(color=color, line=dict(width=0)),
                text=[f"{values[i]:,}" for i in idx],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:,} points<extra></extra>",
            )
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)",
                         title="points visible from this view")
        base_layout(fig, height=max(300, 28 * len(names) + 110),
                    title="Points contributed per view",
                    legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, **FIT)

        with st.expander("Table"):
            show_table(views, hide_index=True)
    else:
        st.info("No per-view statistics yet. Run the projection stage.")

    cov = io.coverage(paths)
    if cov:
        st.caption(
            f"{cov['with_features']:,} of {cov['n_points']:,} points "
            f"({cov['pct']:.1f}%) were seen by at least one view; the rest copied "
            f"a neighbour's feature. Average {cov['mean_views']:.1f} views per point."
        )

    st.divider()
    if views_dir.exists():
        bases = sorted({p.name.rsplit("_", 1)[0]
                        for p in views_dir.glob("*.png")})
        if bases:
            picked = st.selectbox("View", bases)
            cols = st.columns(3)
            for col, suffix, caption in zip(
                cols,
                ["rgb", "features", "depth"],
                ["RGB", "GARField features (PCA of the 256 dims)", "Depth"],
            ):
                img = views_dir / f"{picked}_{suffix}.png"
                if img.exists():
                    show_image(col, str(img), caption)
                else:
                    col.caption(f"{caption}: not saved")
            st.caption(
                "Look at the middle image. If window shapes appear on the facade, "
                "the views are working and the next lever is the side/top balance. "
                "If the facade looks smooth, the GARField scale is too coarse."
            )

            with st.expander(f"All {len(bases)} views at once"):
                kind = st.radio("Show", ["features", "rgb", "depth"],
                                horizontal=True, key="grid_kind")
                per_row = 4
                for start in range(0, len(bases), per_row):
                    row = st.columns(per_row)
                    for col, name in zip(row, bases[start:start + per_row]):
                        img = views_dir / f"{name}_{kind}.png"
                        if img.exists():
                            show_image(col, str(img), name)
        else:
            st.info("No rendered images in the views folder yet.")
    else:
        st.info(f"`{views_dir}` does not exist yet.")


# ----------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------

def bbox_edges(center, scale):
    """Line segments of the crop box, for the 3D plots."""
    c = np.asarray(center, dtype=float)
    s = np.asarray(scale, dtype=float)
    lo, hi = c - s / 2, c + s / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    xs, ys, zs = [], [], []
    for a, b in pairs:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs


with tab_cameras:
    st.subheader("Where the cameras are")
    st.caption(
        "Three camera sets feed three different stages. This is where you see "
        "the angles instead of guessing them."
    )

    center = cfg.get("garfield", {}).get("crop_center", [0, 0, 0])
    scale = cfg.get("garfield", {}).get("crop_scale", [1, 1, 1])

    poses, poses_path = io.load_training_poses(paths)
    ring_views = io.view_camera_positions(paths)

    merge_cfg = cfg.get("merging", {})
    c1, c2, c3, c4 = st.columns(4)
    f_elev = c1.slider("Max elevation", 0.0, 90.0,
                       float(merge_cfg.get("max_elevation_deg", 35.0)), 1.0,
                       help="0 is a horizontal side view, 90 is straight down.")
    f_inc = c2.slider("Max facade incidence", 0.0, 90.0,
                      float(merge_cfg.get("max_incidence_deg", 30.0)), 1.0,
                      help="How far a camera may be from facing a facade head-on.")
    f_off = c3.slider("Max off-axis", 0.0, 90.0,
                      float(merge_cfg.get("max_offaxis_deg", 40.0)), 1.0,
                      help="The camera must point within this angle of the centre.")
    facade_text = c4.text_input(
        "Facade azimuths",
        ", ".join(str(v) for v in merge_cfg.get("facade_azimuths",
                                                [0, 90, 180, 270])),
        help="Directions your facades face. Change these if the building is "
             "not aligned with the X and Y axes.",
    )
    try:
        facade_azimuths = [float(v) for v in facade_text.split(",") if v.strip()]
    except ValueError:
        facade_azimuths = [0, 90, 180, 270]
        st.error("Facade azimuths: use numbers separated by commas")

    if poses is None:
        st.info(
            "`training_camera_poses.npy` was not found, so the merge cameras "
            "cannot be shown. The rings below still work."
        )
        filt = None
        angles = None
    else:
        angles = io.camera_angles(poses, center)
        filt = io.filter_cameras(angles, f_elev, f_inc, f_off, facade_azimuths)

        show_metric_row([
            ("Training cameras", f"{filt['n_total']:,}",
             f"Read from {poses_path.name}."),
            ("Used for merging", f"{filt['n_kept']:,}",
             "Cameras that pass all three angle tests."),
            ("Share kept", f"{100*filt['n_kept']/max(filt['n_total'],1):.0f}%",
             "A small share makes min_support hard to reach."),
            ("Ring views", len(ring_views) if ring_views else 0,
             "Rendered views sent to SAM3."),
        ])

        if filt["n_kept"] < 10:
            st.error(
                f"Only {filt['n_kept']} cameras pass. The merge stage stops "
                "below 10. Widen the angles, or set the facade azimuths to the "
                "real directions of your facades."
            )

        st.markdown("##### Training cameras by angle")
        fig = go.Figure()
        for keep, color, name in ((True, io.SERIES_1, "kept for merging"),
                                  (False, io.SERIES_2, "dropped")):
            sel = filt["keep"] == keep
            if not sel.any():
                continue
            fig.add_scatter(
                x=angles["azimuth"][sel],
                y=angles["elevation"][sel],
                mode="markers",
                name=name,
                marker=dict(size=7, color=color, opacity=0.75),
                text=[f"camera {i}<br>{r}" for i, r in
                      zip(np.where(sel)[0], filt["reason"][sel])],
                hovertemplate="%{text}<br>az %{x:.0f}°, el %{y:.0f}°<extra></extra>",
            )
        for az in facade_azimuths:
            fig.add_vline(x=float(az) % 360, line_width=1, line_dash="dot",
                          line_color=io.MUTED)
        fig.add_hline(y=f_elev, line_width=1, line_dash="dash",
                      line_color=io.MUTED)
        fig.update_xaxes(title="viewing azimuth (degrees)", range=[0, 360],
                         dtick=45, showgrid=True,
                         gridcolor="rgba(128,128,128,0.18)")
        fig.update_yaxes(title="viewing elevation (degrees)", showgrid=True,
                         gridcolor="rgba(128,128,128,0.18)")
        st.plotly_chart(
            base_layout(fig, height=420,
                        title="Each dot is one drone photo",
                        legend=dict(orientation="h", y=1.1, x=0)),
            **FIT,
        )
        st.caption(
            "Dotted vertical lines are your facade directions; the dashed "
            "horizontal line is the elevation limit. Blue dots are the photos "
            "the merge stage is allowed to use."
        )

        counts = {}
        for r in filt["reason"]:
            counts[r] = counts.get(r, 0) + 1
        st.plotly_chart(
            bar_chart(list(counts.keys()), list(counts.values()),
                      "Why cameras were dropped"),
            **FIT,
        )

    st.divider()
    st.markdown("##### The scene in 3D")

    fig = go.Figure()
    xs, ys, zs = bbox_edges(center, scale)
    fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", name="crop box",
                      line=dict(color=io.MUTED, width=3))

    if ring_views:
        pos = np.array([v["position"] for v in ring_views])
        fig.add_scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers", name="ring views (SAM3)",
            marker=dict(size=4, color=io.SERIES_1),
            text=[f"{v['name']}<br>az {v['azimuth']:.0f}°, "
                  f"el {v['elevation']:.0f}°" for v in ring_views],
            hovertemplate="%{text}<extra></extra>",
        )

    proj_stats, _ = io.read_view_stats(paths["proj"])
    ortho = [v for v in (proj_stats or {}).get("views", []) if "plane_center" in v]
    if ortho:
        pos = np.array([v["plane_center"] for v in ortho])
        fig.add_scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers", name="ortho views (features)",
            marker=dict(size=6, color="#1baf7a", symbol="diamond"),
            text=[f"{v['name']}<br>{v['visible_points']:,} points"
                  for v in ortho],
            hovertemplate="%{text}<extra></extra>",
        )

    if poses is not None and filt is not None:
        pos = angles["position"]
        keep = filt["keep"]
        step = max(1, len(pos) // 400)
        for sel, color, name in ((keep, io.SERIES_2, "merge cameras"),
                                 (~keep, "#cfcec7", "other drone photos")):
            idx = np.where(sel)[0][::step]
            if not len(idx):
                continue
            fig.add_scatter3d(
                x=pos[idx, 0], y=pos[idx, 1], z=pos[idx, 2],
                mode="markers", name=name,
                marker=dict(size=2.5, color=color),
                text=[f"camera {i}" for i in idx],
                hovertemplate="%{text}<extra></extra>",
            )

    fig.update_layout(scene=dict(aspectmode="data",
                                 xaxis=dict(showbackground=False, title=""),
                                 yaxis=dict(showbackground=False, title=""),
                                 zaxis=dict(showbackground=False, title="")))
    st.plotly_chart(
        base_layout(fig, height=640, showlegend=True,
                    legend=dict(orientation="h", y=1.05, x=0)),
        **FIT,
    )
    st.caption(
        "Grey box is the building crop. Blue dots are the rendered ring views "
        "that SAM3 sees. Orange dots are the drone photos that the merge stage "
        "is using; pale dots are the ones it skips."
    )


# ----------------------------------------------------------------------
# Clusters
# ----------------------------------------------------------------------

with tab_clusters:
    st.subheader("Clusters in 3D")

    points = io.read_points(paths)
    if points is None:
        st.info("Run the projection stage first.")
    else:
        which = st.radio(
            "Labels",
            ["Fine clusters", "After merging"],
            horizontal=True,
        )
        label_path = (paths["cluster"] / "cluster_labels.npy"
                      if which == "Fine clusters"
                      else paths["merge"] / "merged_labels.npy")
        labels = io.read_labels(label_path)

        if labels is None:
            st.info(f"`{label_path.name}` does not exist yet.")
        elif len(labels) != len(points):
            st.error(
                f"{len(labels):,} labels but {len(points):,} points. "
                "The clustering was run on a different projection output. "
                "Re-run the clustering stage."
            )
        else:
            ids, counts, noise_pct = io.cluster_sizes(labels)
            n_clusters = len(ids)

            c1, c2, c3 = st.columns([1.2, 1.2, 2])
            n_show = c1.slider("Points drawn", 10_000, 200_000, 40_000, 10_000,
                               help="Lower is faster to rotate.")
            hide_noise = c2.checkbox("Hide noise points", value=True)
            highlight = c3.selectbox(
                "Highlight one cluster",
                ["(none)"] + [f"{int(i)}  ·  {int(c):,} pts"
                              for i, c in zip(ids, counts)],
                help="Color separates neighbours; this selector is how you "
                     "identify one cluster.",
            )

            mask = labels >= 0 if hide_noise else np.ones(len(labels), bool)
            idx_all = np.where(mask)[0]
            idx = idx_all[subsample(len(idx_all), n_show)]

            palette = io.cluster_palette(max(n_clusters, 1))
            lab = labels[idx]

            if highlight != "(none)":
                target_id = int(highlight.split("·")[0].strip())
                colors = np.where(lab == target_id, io.SERIES_1, "#d9d9d4")
                hover = [f"cluster {int(l)}" for l in lab]
            else:
                colors = [palette[int(l) % len(palette)] if l >= 0 else "#3a3a38"
                          for l in lab]
                hover = [f"cluster {int(l)}" if l >= 0 else "noise" for l in lab]

            show_metric_row([
                ("Clusters", f"{n_clusters:,}", "Clusters with at least one point."),
                ("Noise", f"{noise_pct:.1f}%", "Points HDBSCAN did not assign."),
                ("Largest cluster", f"{int(counts[0]):,}" if n_clusters else "–",
                 "If one cluster holds most of the cloud, the clustering is too coarse."),
                ("Drawn", f"{len(idx):,}", "Points in the plot below."),
            ])

            if n_clusters and counts[0] / max(counts.sum(), 1) > 0.3:
                st.warning(
                    f"The largest cluster holds "
                    f"{100*counts[0]/counts.sum():.0f}% of the clustered points. "
                    "That is the wall-absorbs-everything problem. A finer "
                    "GARField scale or a higher min-cluster count would help."
                )

            st.plotly_chart(
                point_cloud_figure(points[idx], colors, hover),
            )

            top = min(30, n_clusters)
            if top:
                st.plotly_chart(
                    bar_chart(
                        [f"cluster {int(i)}" for i in ids[:top]],
                        [int(c) for c in counts[:top]],
                        f"Largest {top} clusters, by point count",
                        text=[f"{int(c):,}" for c in counts[:top]],
                    ),
                )

            groups = paths["merge"] / "merged_groups.csv"
            if groups.exists():
                with st.expander("What merging joined"):
                    show_table(
                        [line.split(",") for line in
                         groups.read_text().splitlines()[:200]]
                    )


# ----------------------------------------------------------------------
# Merge evidence
# ----------------------------------------------------------------------

with tab_merge:
    st.subheader("Cluster merging")

    evidence = io.load_evidence(paths)
    candidates = io.load_candidates(paths)
    fine_labels = io.read_labels(paths["cluster"] / "cluster_labels.npy")

    if candidates is None:
        st.info("Run the merge candidate stage first.")
    else:
        cluster_ids = (sorted({int(c) for c in np.unique(fine_labels) if c >= 0})
                       if fine_labels is not None
                       else sorted({r["cluster_a"] for r in candidates}
                                   | {r["cluster_b"] for r in candidates}))

        cfg_support = int(cfg.get("merging", {}).get("min_support", 50))
        cfg_cosine = float(cfg.get("merging", {}).get("min_cosine", 0.90))

        supports = [r["supporting_views"] for r in candidates]
        max_support = max(supports) if supports else 1

        show_metric_row([
            ("Clusters", f"{len(cluster_ids):,}",
             "Fine clusters going into merging."),
            ("Evidence pairs", f"{len(evidence):,}" if evidence else "–",
             "Pairs that ever shared a SAM2 mask."),
            ("Candidate pairs", f"{len(candidates):,}",
             "Pairs that also got a feature similarity score."),
            ("Best support", max_support,
             "Highest number of cameras backing any single pair. If your "
             "min_support is near this, almost nothing can merge."),
        ])

        if cfg_support > max_support:
            st.error(
                f"`min_support` is {cfg_support} but the strongest pair only "
                f"has {max_support} supporting cameras. Nothing can merge."
            )
        elif cfg_support > 0.6 * max_support:
            st.warning(
                f"`min_support` is {cfg_support} and the strongest pair has "
                f"{max_support}. The threshold is close to the ceiling, so "
                "very few pairs can pass. This usually means the camera filter "
                "cut the pool and the threshold was not lowered with it."
            )

        st.markdown("##### Try thresholds without running anything")
        c1, c2 = st.columns(2)
        try_support = safe_slider(c1, "min_support", 1, max(max_support, 2),
                                  min(cfg_support, max_support), 1,
                                  help="Cameras that must agree on a pair.")
        try_cosine = c2.slider("min_cosine", 0.0, 1.0, cfg_cosine, 0.01,
                               help="GARField feature similarity the pair must reach.")

        preview = io.merge_preview(candidates, cluster_ids,
                                   try_support, try_cosine)

        show_metric_row([
            ("Pairs accepted", preview["n_accepted"],
             "Pairs that pass both tests."),
            ("Groups formed", preview["n_groups"],
             "Sets of clusters that become one."),
            ("Clusters after", f"{preview['clusters_after']:,}",
             f"Down from {preview['clusters_before']:,}."),
            ("Chained groups", preview["chained"],
             "Groups holding more than two clusters — merging already chains "
             "through shared members."),
        ])

        if preview["largest_group"]:
            st.caption(
                f"Largest group joins {preview['largest_group']} clusters. "
                "Merging uses union-find, so if 1+2 and 2+3 both pass you get "
                "one group {1, 2, 3}. What it does **not** do is re-measure the "
                "merged group against a further cluster with fresh evidence — "
                "every decision comes from the original pairwise numbers."
            )

        fig = go.Figure()
        acc = [(r["supporting_views"], r["cosine_similarity"]) for r in candidates
               if r["supporting_views"] >= try_support
               and r["cosine_similarity"] >= try_cosine]
        rej = [(r["supporting_views"], r["cosine_similarity"]) for r in candidates
               if not (r["supporting_views"] >= try_support
                       and r["cosine_similarity"] >= try_cosine)]
        for data, color, name in ((rej, "#cfcec7", "rejected"),
                                  (acc, io.SERIES_1, "accepted")):
            if not data:
                continue
            fig.add_scatter(
                x=[d[0] for d in data], y=[d[1] for d in data],
                mode="markers", name=name,
                marker=dict(size=8, color=color, opacity=0.8),
                hovertemplate="%{x} views · cosine %{y:.3f}<extra></extra>",
            )
        fig.add_vline(x=try_support, line_width=1, line_dash="dash",
                      line_color=io.MUTED)
        fig.add_hline(y=try_cosine, line_width=1, line_dash="dash",
                      line_color=io.MUTED)
        fig.update_xaxes(title="supporting cameras", showgrid=True,
                         gridcolor="rgba(128,128,128,0.18)")
        fig.update_yaxes(title="feature cosine similarity", showgrid=True,
                         gridcolor="rgba(128,128,128,0.18)")
        st.plotly_chart(
            base_layout(fig, height=440, title="Every candidate pair",
                        legend=dict(orientation="h", y=1.1, x=0)),
            **FIT,
        )
        st.caption(
            "Everything up and to the right of the dashed lines merges. "
            "Drag the sliders to see how many pairs that is."
        )

        if preview["groups"]:
            st.markdown("##### Groups this setting would create")
            show_table(
                [{"size": len(g), "clusters": ", ".join(str(c) for c in g)}
                 for g in preview["groups"][:200]],
                hide_index=True,
            )

        with st.expander("All candidate pairs"):
            show_table(sorted(candidates,
                               key=lambda r: -r["supporting_views"])[:500],
                       hide_index=True)

    st.divider()
    st.markdown("##### The images the merge evidence comes from")
    st.caption(
        "Merging is decided on the SAM2 masks cached for your drone photos, "
        "not on the rendered views. Each mask groups pixels; two clusters that "
        "land inside the same mask, in enough cameras, become one."
    )

    cache_files = io.list_sam_cache(cfg)
    if not cache_files:
        st.info(f"No `sam_*.npz` files in `{io.sam_cache_dir(cfg)}`.")
    else:
        images = io.dataset_images(cfg)
        c1, c2 = st.columns([2, 1])
        idx = safe_slider(c1, "Camera index", 0, len(cache_files) - 1, 0, 1,
                          help="Matches sam_XXXXXX.npz, which follows the "
                               "training camera order.")
        npz_path = cache_files[idx]

        data = np.load(npz_path)
        if "pixel_level_keys" not in data:
            st.warning(f"`{npz_path.name}` has no `pixel_level_keys`.")
        else:
            keys = data["pixel_level_keys"]
            n_levels = keys.shape[2] if keys.ndim == 3 else 1
            if keys.ndim == 2:
                keys = keys[:, :, None]
            level = safe_slider(c2, "Grouping level", 0, n_levels - 1, 0, 1,
                                help="SAM2 groups at several scales. Level 0 "
                                     "is the finest.")
            n_groups = int(len(np.unique(keys[:, :, level])) - 1)
            st.caption(f"`{npz_path.name}` · {keys.shape[1]}x{keys.shape[0]} "
                       f"pixels · {keys.shape[2]} levels · "
                       f"{n_groups} groups at this level")

            col1, col2 = st.columns(2)
            show_image(col1, io.colorize_groups(keys[:, :, level]),
                       f"SAM2 groups, level {level}")
            if idx < len(images):
                show_image(col2, str(images[idx]),
                           f"{images[idx].name} (assuming the cache "
                           "follows sorted image order)")
            else:
                col2.info("No matching image found in the dataset folder.")

            if poses is not None and filt is not None and idx < len(filt["keep"]):
                used = filt["keep"][idx]
                st.caption(
                    ("✅ This camera **is** used as merge evidence."
                     if used else
                     f"⬜ This camera is **not** used: {filt['reason'][idx]}.")
                )


# ----------------------------------------------------------------------
# Semantic result
# ----------------------------------------------------------------------

with tab_semantic:
    st.subheader("Semantic result")

    rows = io.semantic_table(paths)
    points = io.read_points(paths)
    labels = io.read_labels(paths["merge"] / "merged_labels.npy")

    if not rows:
        st.info("Run the semantic labeling stage first.")
    else:
        totals = io.label_totals(rows)
        total_points = sum(totals.values()) or 1

        st.plotly_chart(
            bar_chart(
                list(totals.keys()),
                list(totals.values()),
                "Points per class",
                text=[f"{v:,}  ({100*v/total_points:.1f}%)"
                      for v in totals.values()],
            ),
        )

        unknown = 100 * totals.get("unknown", 0) / total_points
        window = 100 * totals.get("window", 0) / total_points
        if unknown > 20:
            st.warning(
                f"{unknown:.0f}% of points are `unknown` — those clusters got no "
                "mask vote above the IoU threshold. Lower **Min IoU** in Settings, "
                "or check that the SAM3 masks actually cover these views."
            )
        if window < 2:
            st.warning(
                f"Only {window:.1f}% of points are `window`. Windows are still "
                "being absorbed into wall clusters."
            )

        if points is not None and labels is not None and len(labels) == len(points):
            class_of = {r["cluster"]: r["label"] for r in rows}
            classes = sorted(totals.keys(), key=lambda c: -totals[c])
            chosen = st.multiselect("Classes to show", classes, default=classes)

            n_show = st.slider("Points drawn", 10_000, 200_000, 40_000, 10_000,
                               key="sem_points")

            label_names = np.array(
                [class_of.get(int(l), "noise") if l >= 0 else "noise"
                 for l in labels]
            )
            keep = np.isin(label_names, chosen)
            idx_all = np.where(keep)[0]
            if len(idx_all):
                idx = idx_all[subsample(len(idx_all), n_show)]
                colors = [io.CLASS_COLORS.get(n, "#4d4d4d") for n in label_names[idx]]
                hover = [f"{n} · cluster {int(labels[i])}"
                         for n, i in zip(label_names[idx], idx)]
                st.plotly_chart(
                    point_cloud_figure(points[idx], colors, hover),
                )
                st.caption(
                    "Colors are the same ones written into the PLY files, so this "
                    "matches what you see in CloudCompare. Class identity comes "
                    "from the filter above and the hover text."
                )
            else:
                st.info("No points in the selected classes.")

        st.markdown("#### Per-cluster votes")
        only_unknown = st.checkbox("Only clusters labeled unknown", value=False)
        table = [r for r in rows if not only_unknown or r["label"] == "unknown"]
        show_table(table, hide_index=True)


# ----------------------------------------------------------------------
# SAM3 masks
# ----------------------------------------------------------------------

with tab_masks:
    st.subheader("SAM3 masks on the rendered views")

    masks_dir = paths["masks"]
    views_dir = paths["views"]

    npz_files = sorted(masks_dir.glob("*.npz")) if masks_dir.exists() else []
    if not npz_files:
        st.info(f"No mask files in `{masks_dir}` yet. Run the SAM3 stage.")
    else:
        names = [p.stem.replace("_masks", "") for p in npz_files]
        picked = st.selectbox("View", names)
        npz_path = npz_files[names.index(picked)]

        min_score = st.slider("Minimum mask score", 0.0, 1.0, 0.3, 0.05)

        data = np.load(npz_path)
        classes = sorted({k[:-6] for k in data.files if k.endswith("_masks")})

        counts = {}
        for cls in classes:
            scores = data.get(f"{cls}_scores")
            n = int((np.asarray(scores) >= min_score).sum()) if scores is not None \
                else len(data[f"{cls}_masks"])
            if n:
                counts[cls.replace("_", " ")] = n

        if counts:
            st.plotly_chart(
                bar_chart(list(counts.keys()), list(counts.values()),
                          "Masks found in this view"),
            )
        else:
            st.warning("No masks above this score in this view.")

        image = views_dir / f"{picked}.jpg"
        col1, col2 = st.columns(2)
        if image.exists():
            show_image(col1, str(image), "rendered view")
        else:
            col1.info(f"`{image.name}` not found")

        pick_class = col2.selectbox("Draw this class", ["(none)"] +
                                    list(counts.keys()))
        if pick_class != "(none)" and image.exists():
            try:
                from PIL import Image

                base = Image.open(image).convert("RGB")
                arr = np.array(base).astype(np.float32)
                key = pick_class.replace(" ", "_")
                masks = np.asarray(data[f"{key}_masks"])
                scores = np.asarray(data.get(f"{key}_scores",
                                             np.ones(len(masks))))
                tint = np.array([0, 204, 255], dtype=np.float32)
                drawn = 0
                for m, s in zip(masks, scores):
                    if s < min_score:
                        continue
                    m = np.asarray(m).astype(bool)
                    if m.shape[:2] != arr.shape[:2]:
                        continue
                    arr[m] = 0.55 * arr[m] + 0.45 * tint
                    drawn += 1
                show_image(col2, arr.astype(np.uint8),
                           f"{drawn} `{pick_class}` masks")
            except Exception as exc:  # pragma: no cover
                col2.error(f"Could not draw the masks: {exc}")


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------

with tab_history:
    st.subheader("Run history")
    st.caption(
        "Every run started from this page is recorded with the settings that "
        "were active and the numbers that came out. This is how you see which "
        "variable actually moved the result."
    )

    history = io.load_history()
    if not history:
        st.info("No runs recorded yet. Start a stage from the Run tab.")
    else:
        show_table(list(reversed(history)), hide_index=True)

        numeric = sorted({
            k for row in history for k, v in row.items()
            if isinstance(v, (int, float)) and k not in ("exit_code",)
        })
        if numeric:
            c1, c2 = st.columns(2)
            metric = c1.selectbox("Metric", numeric,
                                  index=numeric.index("n_clusters")
                                  if "n_clusters" in numeric else 0)
            against = c2.selectbox("Against", ["run order"] + numeric)

            xs, ys, texts = [], [], []
            for i, row in enumerate(history):
                if metric not in row or row[metric] is None:
                    continue
                if against == "run order":
                    xs.append(i + 1)
                elif isinstance(row.get(against), (int, float)):
                    xs.append(row[against])
                else:
                    continue
                ys.append(row[metric])
                texts.append(f"{row['time']} · {row['stage']}")

            if xs:
                fig = go.Figure(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        line=dict(color=io.SERIES_1, width=2),
                        marker=dict(size=9, color=io.SERIES_1),
                        text=texts,
                        hovertemplate="%{text}<br>%{y}<extra></extra>",
                    )
                )
                fig.update_xaxes(title=against, showgrid=True,
                                 gridcolor="rgba(128,128,128,0.18)")
                fig.update_yaxes(title=metric, showgrid=True,
                                 gridcolor="rgba(128,128,128,0.18)")
                st.plotly_chart(base_layout(fig, title=f"{metric} across runs"),
                                **FIT)

        if st.button("Clear history"):
            io.HISTORY_PATH.unlink(missing_ok=True)
            st.rerun()
