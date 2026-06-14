#!/usr/bin/env python3
"""
Pipeline Dashboard Generator

Scans all pipeline outputs and generates an interactive HTML dashboard.
Open the HTML file in a browser to explore results.

Usage:
    conda activate nerfstudio3
    python pipeline/generate_dashboard.py
    # Opens outputs/pipeline_report/dashboard.html
"""

import numpy as np
import json
import os
import glob
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import base64

try:
    import yaml
    with open("pipeline/config.yaml") as f:
        cfg = yaml.safe_load(f)
except:
    cfg = {}

REPORT_DIR = Path("outputs/pipeline_report")


def collect_pipeline_data():
    """Collect all available data from pipeline outputs."""
    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }

    # ── Stage 1: COLMAP ──
    dataset_path = Path(cfg.get("dataset", {}).get("path", "data/PFTdrone"))
    images_dir = dataset_path / "images"
    sparse_dir = dataset_path / "sparse" / "0"

    stage1 = {"name": "COLMAP", "status": "pending", "data": {}}
    if images_dir.exists():
        imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
               list(images_dir.glob("*.png"))
        stage1["data"]["n_images"] = len(imgs)
    if sparse_dir.exists():
        stage1["status"] = "complete"
        for f in ["cameras.bin", "images.bin", "points3D.bin"]:
            p = sparse_dir / f
            if p.exists():
                stage1["data"][f] = round(p.stat().st_size / (1024*1024), 1)
    data["stages"]["colmap"] = stage1

    # ── Stage 2a: GARField ──
    stage2a = {"name": "GARField (NeRF)", "status": "pending", "data": {}}
    configs = sorted(glob.glob("outputs/*/garfield/*/config.yml"),
                     key=os.path.getmtime, reverse=True)
    if configs:
        stage2a["status"] = "complete"
        stage2a["data"]["config"] = configs[0]
    data["stages"]["garfield"] = stage2a

    # ── Stage 2b: GARField-Gauss ──
    stage2b = {"name": "GARField-Gauss", "status": "pending", "data": {}}
    configs = sorted(glob.glob("outputs/*/garfield-gauss/*/config.yml"),
                     key=os.path.getmtime, reverse=True)
    if configs:
        stage2b["status"] = "complete"
        stage2b["data"]["config"] = configs[0]
    data["stages"]["garfield_gauss"] = stage2b

    # ── Stage 3: Orthographic Projection ──
    stage3 = {"name": "Ortho Projection", "status": "pending", "data": {}}
    for proj_dir in ["outputs/ortho_projection_s005", "outputs/ortho_projection_cropped",
                     "outputs/ortho_projection"]:
        feat_path = Path(proj_dir) / "avg_features.npy"
        pts_path = Path(proj_dir) / "points.npy"
        if feat_path.exists() and pts_path.exists():
            features = np.load(feat_path)
            points = np.load(pts_path)
            stage3["status"] = "complete"
            stage3["data"] = {
                "n_points": len(points),
                "feature_dim": int(features.shape[1]),
                "feature_mean": round(float(features.mean()), 4),
                "feature_std": round(float(features.std()), 4),
                "bounds": {
                    "x": [round(float(points[:,0].min()),3), round(float(points[:,0].max()),3)],
                    "y": [round(float(points[:,1].min()),3), round(float(points[:,1].max()),3)],
                    "z": [round(float(points[:,2].min()),3), round(float(points[:,2].max()),3)],
                },
                "feature_magnitude_hist": np.histogram(
                    np.linalg.norm(features, axis=1), bins=30
                )[0].tolist(),
            }
            break
    data["stages"]["projection"] = stage3

    # ── Stage 4: Clustering ──
    stage4 = {"name": "HDBSCAN Clustering", "status": "pending", "data": {}}
    for labels_path in ["outputs/ortho_projection_cropped/cluster_labels.npy",
                        "outputs/ortho_projection_s005/cluster_labels.npy"]:
        if Path(labels_path).exists():
            labels = np.load(labels_path)
            n_clusters = int(labels.max()) + 1
            noise = int((labels == -1).sum())
            sizes = [int((labels == i).sum()) for i in range(n_clusters)]
            stage4["status"] = "complete"
            stage4["data"] = {
                "n_points": len(labels),
                "n_clusters": n_clusters,
                "noise_points": noise,
                "noise_pct": round(100 * noise / len(labels), 1),
                "cluster_sizes": sorted(sizes, reverse=True),
                "size_min": min(sizes),
                "size_max": max(sizes),
                "size_mean": round(float(np.mean(sizes)), 0),
                "size_median": round(float(np.median(sizes)), 0),
            }
            break
    data["stages"]["clustering"] = stage4

    # ── Stage 5: Render Views ──
    stage5 = {"name": "Render Views", "status": "pending", "data": {}}
    views_dir = Path(cfg.get("labeling_views", {}).get("output_dir", "outputs/labeling_views"))
    params_path = views_dir / "view_params.json"
    if params_path.exists():
        with open(params_path) as f:
            vp = json.load(f)
        imgs = list(views_dir.glob("*.jpg")) + list(views_dir.glob("*.png"))
        stage5["status"] = "complete"

        elevations = defaultdict(int)
        for v in vp:
            for p in v['view_name'].split('_'):
                if p.startswith('el'):
                    elevations[p[2:] + "°"] += 1

        # Encode thumbnails (first 6)
        thumbnails = []
        for img_path in sorted(imgs)[:6]:
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                thumbnails.append({
                    "name": img_path.stem,
                    "data": f"data:image/jpeg;base64,{b64}",
                })
            except:
                pass

        stage5["data"] = {
            "n_views": len(vp),
            "n_images": len(imgs),
            "elevations": dict(elevations),
            "thumbnails": thumbnails,
        }
    data["stages"]["render_views"] = stage5

    # ── Stage 6: SAM 3 Inference ──
    stage6 = {"name": "SAM 3 PCS Inference", "status": "pending", "data": {}}
    for masks_dir in ["outputs/labeling_masks_finetuned",
                      "garfield/outputs/labeling_masks_finetuned"]:
        masks_dir = Path(masks_dir)
        npz_files = sorted(masks_dir.glob("*.npz")) if masks_dir.exists() else []
        if npz_files:
            total = defaultdict(int)
            per_view = {}
            for npz_path in npz_files:
                view_name = npz_path.stem.replace("_masks", "")
                d = np.load(npz_path)
                view_counts = {}
                for key in d.files:
                    if key.endswith("_masks"):
                        label = key[:-6].replace("_", " ")
                        count = len(d[key])
                        total[label] += count
                        view_counts[label] = count
                per_view[view_name] = view_counts

            stage6["status"] = "complete"
            stage6["data"] = {
                "n_files": len(npz_files),
                "total_detections": dict(sorted(total.items(), key=lambda x: -x[1])),
                "total_count": sum(total.values()),
                "per_view": per_view,
            }
            break
    data["stages"]["sam3"] = stage6

    # ── Stage 7: Semantic Labeling ──
    stage7 = {"name": "Semantic Labeling", "status": "pending", "data": {}}
    for output_dir in ["outputs/semantic_labels_finetuned", "outputs/semantic_labels_final"]:
        labels_path = Path(output_dir) / "semantic_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                results = json.load(f)

            label_summary = defaultdict(int)
            total_pts = 0
            for cid, r in results.items():
                label_summary[r['label']] += r['n_points']
                total_pts += r['n_points']

            label_pcts = {l: round(100*c/total_pts, 1)
                         for l, c in sorted(label_summary.items(), key=lambda x: -x[1])}

            stage7["status"] = "complete"
            stage7["data"] = {
                "n_clusters": len(results),
                "total_points": total_pts,
                "label_distribution": label_pcts,
                "label_counts": dict(sorted(label_summary.items(), key=lambda x: -x[1])),
                "baseline": {"roof": 62.1, "ground": 1.2, "unknown": 36.7},
            }

            ply_path = Path(output_dir) / "semantic_pointcloud.ply"
            if ply_path.exists():
                stage7["data"]["ply_size_mb"] = round(ply_path.stat().st_size / (1024*1024), 1)
            break
    data["stages"]["semantic"] = stage7

    return data


def generate_html(data):
    """Generate interactive HTML dashboard."""

    stages_json = json.dumps(data, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drone-to-BIM Pipeline Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f1117; color: #e1e1e6; padding: 20px; }}
h1 {{ font-size: 24px; margin-bottom: 8px; color: #fff; }}
h2 {{ font-size: 18px; margin-bottom: 12px; color: #a5a5b5; }}
h3 {{ font-size: 15px; margin-bottom: 8px; color: #c5c5d5; }}
.subtitle {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
.grid {{ display: grid; gap: 16px; margin-bottom: 24px; }}
.grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
.card {{ background: #1a1b26; border-radius: 12px; padding: 20px; border: 1px solid #2a2b36; }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
.stat {{ text-align: center; }}
.stat-value {{ font-size: 28px; font-weight: 600; color: #fff; }}
.stat-label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.badge-complete {{ background: #1a3a2a; color: #4ade80; }}
.badge-pending {{ background: #3a2a1a; color: #fbbf24; }}
.pipeline-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }}
.pipeline-step {{ flex: 1; min-width: 100px; padding: 12px; border-radius: 8px; text-align: center;
                  font-size: 12px; border: 1px solid #2a2b36; cursor: pointer; transition: all 0.2s; }}
.pipeline-step:hover {{ transform: translateY(-2px); }}
.pipeline-step.complete {{ background: #1a2a1a; border-color: #4ade80; }}
.pipeline-step.pending {{ background: #2a2a1a; border-color: #fbbf24; }}
.pipeline-step .step-name {{ font-weight: 500; color: #fff; margin-top: 4px; }}
.pipeline-step .step-icon {{ font-size: 18px; }}
.chart-container {{ position: relative; height: 280px; }}
.thumbs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.thumbs img {{ width: 120px; height: 80px; object-fit: cover; border-radius: 6px; border: 1px solid #2a2b36; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #2a2b36; color: #888; font-weight: 500; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #1a1b26; }}
tr:hover {{ background: #1f2030; }}
.tab-bar {{ display: flex; gap: 4px; margin-bottom: 16px; }}
.tab {{ padding: 8px 16px; border-radius: 8px 8px 0 0; cursor: pointer; font-size: 13px;
        background: #1a1b26; color: #888; border: 1px solid #2a2b36; border-bottom: none; }}
.tab.active {{ background: #2a2b36; color: #fff; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.comparison-row {{ display: flex; align-items: center; margin: 6px 0; }}
.comparison-label {{ width: 100px; font-size: 13px; color: #888; }}
.comparison-bars {{ flex: 1; display: flex; gap: 4px; flex-direction: column; }}
.bar-row {{ display: flex; align-items: center; gap: 8px; }}
.bar {{ height: 16px; border-radius: 4px; min-width: 2px; transition: width 0.5s; }}
.bar-before {{ background: #666; }}
.bar-after {{ background: #7c3aed; }}
.bar-value {{ font-size: 11px; color: #888; min-width: 40px; }}
</style>
</head>
<body>

<h1>Drone-to-BIM Pipeline Dashboard</h1>
<p class="subtitle">Generated: {data['generated_at']}</p>

<div class="pipeline-row" id="pipelineSteps"></div>

<div class="grid grid-4" id="summaryCards"></div>

<div class="grid grid-2">
  <div class="card" id="clusterCard">
    <h3>Cluster Size Distribution</h3>
    <div class="chart-container"><canvas id="clusterChart"></canvas></div>
  </div>
  <div class="card" id="detectionCard">
    <h3>SAM 3 Detections by Class</h3>
    <div class="chart-container"><canvas id="detectionChart"></canvas></div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card" id="comparisonCard">
    <h3>Before vs After Fine-tuning</h3>
    <div class="chart-container"><canvas id="comparisonChart"></canvas></div>
  </div>
  <div class="card" id="heatmapCard">
    <h3>Detections per View</h3>
    <div class="chart-container"><canvas id="heatmapChart"></canvas></div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card" id="semanticCard">
    <h3>Semantic Label Distribution</h3>
    <div class="chart-container"><canvas id="pieChart"></canvas></div>
  </div>
  <div class="card" id="viewsCard">
    <h3>Rendered Views</h3>
    <div class="thumbs" id="viewThumbs"></div>
  </div>
</div>

<div class="card" style="margin-top:16px" id="detailsCard">
  <h3>Stage Details</h3>
  <table id="detailsTable">
    <thead><tr><th>Stage</th><th>Status</th><th>Key Metrics</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const DATA = {stages_json};
const stages = DATA.stages;

// Color palette
const COLORS = {{
  purple: '#7c3aed', teal: '#14b8a6', coral: '#f97316', green: '#22c55e',
  red: '#ef4444', blue: '#3b82f6', gray: '#6b7280', pink: '#ec4899',
  yellow: '#eab308', cyan: '#06b6d4'
}};
const LABEL_COLORS = {{
  wall: '#B3B3A0', roof: '#CC3333', vegetation: '#009900', opening: '#FF9900',
  ground: '#666666', unknown: '#4D4D4D', window: '#00CCff', door: '#CC6600',
  column: '#E6E6E6', beam: '#996600', ceiling: '#CCCCDD', sky: '#80B3FF'
}};

// Pipeline steps
const stepDefs = [
  {{ key: 'colmap', icon: '📷', name: 'COLMAP' }},
  {{ key: 'garfield', icon: '🧠', name: 'GARField' }},
  {{ key: 'garfield_gauss', icon: '💠', name: 'Gauss-Splat' }},
  {{ key: 'projection', icon: '📐', name: 'Projection' }},
  {{ key: 'clustering', icon: '🔮', name: 'Clustering' }},
  {{ key: 'render_views', icon: '🎬', name: 'Render' }},
  {{ key: 'sam3', icon: '🔍', name: 'SAM 3' }},
  {{ key: 'semantic', icon: '🏷️', name: 'Labeling' }},
];

// Render pipeline steps
const stepsEl = document.getElementById('pipelineSteps');
stepDefs.forEach(s => {{
  const st = stages[s.key];
  const cls = st && st.status === 'complete' ? 'complete' : 'pending';
  stepsEl.innerHTML += `<div class="pipeline-step ${{cls}}">
    <div class="step-icon">${{s.icon}}</div>
    <div class="step-name">${{s.name}}</div>
    <span class="badge badge-${{cls}}">${{cls}}</span>
  </div>`;
}});

// Summary cards
const cardsEl = document.getElementById('summaryCards');
const summaryItems = [];
if (stages.colmap?.data?.n_images) summaryItems.push({{ v: stages.colmap.data.n_images, l: 'Input Images' }});
if (stages.projection?.data?.n_points) summaryItems.push({{ v: stages.projection.data.n_points.toLocaleString(), l: 'Points' }});
if (stages.clustering?.data?.n_clusters) summaryItems.push({{ v: stages.clustering.data.n_clusters, l: 'Clusters' }});
if (stages.sam3?.data?.total_count) summaryItems.push({{ v: stages.sam3.data.total_count, l: 'Detections' }});
if (stages.semantic?.data?.total_points) summaryItems.push({{ v: stages.semantic.data.total_points.toLocaleString(), l: 'Labeled Points' }});
const completedCount = stepDefs.filter(s => stages[s.key]?.status === 'complete').length;
summaryItems.push({{ v: `${{completedCount}}/${{stepDefs.length}}`, l: 'Stages Complete' }});

summaryItems.forEach(item => {{
  cardsEl.innerHTML += `<div class="card stat">
    <div class="stat-value">${{item.v}}</div>
    <div class="stat-label">${{item.l}}</div>
  </div>`;
}});

// Cluster chart
if (stages.clustering?.data?.cluster_sizes) {{
  new Chart(document.getElementById('clusterChart'), {{
    type: 'bar',
    data: {{
      labels: stages.clustering.data.cluster_sizes.map((_, i) => i+1),
      datasets: [{{ data: stages.clustering.data.cluster_sizes, backgroundColor: COLORS.purple,
                    borderRadius: 3, borderSkipped: false }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#1a1b26' }}, ticks: {{ color: '#666', font: {{ size: 9 }} }} }},
        y: {{ grid: {{ color: '#2a2b36' }}, ticks: {{ color: '#666' }} }}
      }}
    }}
  }});
}}

// Detection chart
if (stages.sam3?.data?.total_detections) {{
  const det = stages.sam3.data.total_detections;
  const detLabels = Object.keys(det);
  const detValues = Object.values(det);
  new Chart(document.getElementById('detectionChart'), {{
    type: 'bar',
    data: {{
      labels: detLabels,
      datasets: [{{ data: detValues,
                    backgroundColor: detLabels.map(l => LABEL_COLORS[l] || COLORS.gray),
                    borderRadius: 3, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#2a2b36' }}, ticks: {{ color: '#666' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#ccc', font: {{ size: 11 }} }} }}
      }}
    }}
  }});
}}

// Comparison chart
if (stages.semantic?.data?.label_distribution) {{
  const baseline = stages.semantic.data.baseline || {{}};
  const finetuned = stages.semantic.data.label_distribution;
  const allLabels = [...new Set([...Object.keys(baseline), ...Object.keys(finetuned)])];
  allLabels.sort((a, b) => (finetuned[b]||0) - (finetuned[a]||0));

  new Chart(document.getElementById('comparisonChart'), {{
    type: 'bar',
    data: {{
      labels: allLabels,
      datasets: [
        {{ label: 'Off-the-shelf', data: allLabels.map(l => baseline[l]||0),
           backgroundColor: '#666', borderRadius: 3 }},
        {{ label: 'Fine-tuned', data: allLabels.map(l => finetuned[l]||0),
           backgroundColor: COLORS.purple, borderRadius: 3 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#888' }} }} }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ color: '#888', font: {{ size: 10 }}, maxRotation: 45 }} }},
        y: {{ title: {{ display: true, text: '% of points', color: '#888' }},
              grid: {{ color: '#2a2b36' }}, ticks: {{ color: '#666' }} }}
      }}
    }}
  }});
}}

// Heatmap (detections per view)
if (stages.sam3?.data?.per_view) {{
  const perView = stages.sam3.data.per_view;
  const views = Object.keys(perView).sort();
  const topLabels = Object.keys(stages.sam3.data.total_detections).slice(0, 6);
  const heatData = [];
  views.forEach((v, vi) => {{
    topLabels.forEach((l, li) => {{
      heatData.push({{ x: vi, y: li, v: perView[v][l] || 0 }});
    }});
  }});
  const maxVal = Math.max(...heatData.map(d => d.v));

  new Chart(document.getElementById('heatmapChart'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        data: heatData.map(d => ({{ x: d.x, y: d.y }})),
        pointRadius: heatData.map(d => Math.max(2, 12 * d.v / (maxVal||1))),
        backgroundColor: heatData.map(d => `rgba(124,58,237,${{0.2 + 0.8*d.v/(maxVal||1)}})`),
        pointStyle: 'circle'
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ type: 'linear', min: -0.5, max: views.length-0.5,
              ticks: {{ callback: i => views[i]?.split('_')[1] || '', color: '#666', font: {{ size: 8 }} }},
              grid: {{ color: '#1a1b26' }} }},
        y: {{ type: 'linear', min: -0.5, max: topLabels.length-0.5,
              ticks: {{ callback: i => topLabels[i] || '', color: '#888', font: {{ size: 10 }} }},
              grid: {{ color: '#1a1b26' }} }}
      }}
    }}
  }});
}}

// Pie chart
if (stages.semantic?.data?.label_distribution) {{
  const ld = stages.semantic.data.label_distribution;
  new Chart(document.getElementById('pieChart'), {{
    type: 'doughnut',
    data: {{
      labels: Object.keys(ld),
      datasets: [{{
        data: Object.values(ld),
        backgroundColor: Object.keys(ld).map(l => LABEL_COLORS[l] || COLORS.gray),
        borderColor: '#1a1b26', borderWidth: 2
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ color: '#888', padding: 12, font: {{ size: 11 }} }} }} }}
    }}
  }});
}}

// View thumbnails
if (stages.render_views?.data?.thumbnails) {{
  const thumbsEl = document.getElementById('viewThumbs');
  stages.render_views.data.thumbnails.forEach(t => {{
    thumbsEl.innerHTML += `<img src="${{t.data}}" title="${{t.name}}" alt="${{t.name}}">`;
  }});
}}

// Details table
const tbody = document.querySelector('#detailsTable tbody');
stepDefs.forEach(s => {{
  const st = stages[s.key];
  if (!st) return;
  const status = st.status === 'complete'
    ? '<span class="badge badge-complete">COMPLETE</span>'
    : '<span class="badge badge-pending">PENDING</span>';
  let metrics = '';
  const d = st.data || {{}};
  if (s.key === 'colmap') metrics = `${{d.n_images || '?'}} images`;
  if (s.key === 'garfield') metrics = d.config || '-';
  if (s.key === 'garfield_gauss') metrics = d.config || '-';
  if (s.key === 'projection') metrics = `${{d.n_points?.toLocaleString() || '?'}} points, ${{d.feature_dim || '?'}}D features`;
  if (s.key === 'clustering') metrics = `${{d.n_clusters || '?'}} clusters, ${{d.noise_pct || '?'}}% noise`;
  if (s.key === 'render_views') metrics = `${{d.n_views || '?'}} views`;
  if (s.key === 'sam3') metrics = `${{d.total_count || '?'}} detections across ${{d.n_files || '?'}} views`;
  if (s.key === 'semantic') metrics = Object.entries(d.label_distribution || {{}}).map(([l,p]) => `${{l}}: ${{p}}%`).join(', ');

  tbody.innerHTML += `<tr><td>${{s.icon}} ${{s.name}}</td><td>${{status}}</td><td style="font-size:12px">${{metrics}}</td></tr>`;
}});
</script>
</body>
</html>"""

    return html


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting pipeline data...")
    data = collect_pipeline_data()

    print("Generating dashboard...")
    html = generate_html(data)

    output_path = REPORT_DIR / "dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard saved: {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
