# Pipeline control panel

A small web page for running the drone-to-BIM pipeline and looking at what
comes out of each stage, instead of typing snakemake commands.

## Install (once)

```bash
pip install "protobuf==4.25.7" "streamlit>=1.31,<1.40" plotly ruamel.yaml
```

**Pin those versions.** Streamlit 1.50 and newer require protobuf 5.26 or
above, but nerfstudio requires protobuf 5 or below. Installing the newest
Streamlit upgrades protobuf and breaks `pyarrow`, `nerfstudio` and `wandb`
in that environment. Streamlit 1.3x works with protobuf 4 and the page
supports it.

If you would rather have the newest Streamlit, put it in its own
environment instead, and set **Command prefix** in the sidebar to
`conda run -n base` so snakemake still runs from the environment that has it.

`ruamel.yaml` is optional. Without it the page still saves `config.yaml`,
but your comments in that file are lost. A timestamped backup is written
before every save either way.

The page also survives a broken `pyarrow`: tables fall back to plain text
and the ring editor to a text box, instead of the whole page failing.

## Start

Always start it from the repository root, not from inside `pipeline/`:

```bash
cd ~/garfield
streamlit run pipeline/ui/app.py
```

Then open <http://localhost:8501> in your browser on Windows. WSL forwards
the port for you, so no extra setup is needed.

To stop it, press `Ctrl+C` in the terminal.

## The tabs

**Overview** — how far the pipeline has got, with the numbers that matter:
point count, view coverage, the side/top view balance, cluster count, noise,
and the share of points that ended up `unknown`.

**Settings** — every value that changes the result, with an explanation on
each one. It writes into `pipeline/config.yaml`. Nothing is saved until you
press the save button.

**Run** — one *Check* and one *Run* button per stage. *Check* is a dry run
and changes nothing. Every command is built as

```
snakemake -s pipeline/Snakefile --cores 1 --rerun-triggers mtime \
    --allowed-rules <stage> --force <target file>
```

so only the stage you picked can run. GARField training is never touched.
The log appears on the page while it runs and is also saved under
`outputs/ui/logs/`.

**Projection views** — the rendered views with their RGB image, the PCA of
the GARField features, and the depth map. Plus a bar chart of how many
points each view contributed. This is where you see whether roof views are
drowning out the facade views.

**Cameras** — all three camera sets in one place. A scatter of every drone
photo by viewing azimuth and elevation, colored by whether the merge stage
is allowed to use it, with the filter angles as live sliders. A bar chart of
why cameras were dropped. And a 3D scene with the crop box, the rendered
ring views, the orthographic view positions, and the drone photos.

**Merge evidence** — the candidate pairs plotted as supporting cameras
against feature similarity, with `min_support` and `min_cosine` as sliders
that recompute the merge **on the page**, using the same union-find the
pipeline uses. So you can see how many clusters would survive before running
anything. It warns when `min_support` is close to the strongest pair, which
is the state where almost nothing can merge. Below that, the SAM2 cache
images the evidence is actually built from, level by level.

**Clusters** — the clustered point cloud in 3D, before and after merging.
Colors separate neighbouring clusters; the *Highlight one cluster* selector
and the hover text are what identify a specific cluster.

**Semantic result** — the final labeled cloud, filtered by class, using the
same colors that go into the PLY files. Plus the per-cluster vote table, so
you can see exactly why a cluster got its label.

**SAM3 masks** — the ring views with the SAM3 masks drawn on top, per class,
with a score threshold.

**History** — every run started from this page, with the settings that were
active and the numbers that came out. Use the chart at the bottom to plot
one metric against another and see which variable actually moved the result.

## Notes

- The page reads the same `config.yaml` the Snakefile reads, and rebuilds
  paths the same way, so what you see is what the pipeline will use.
- `sam3_inference` opens SSH to the HPC. If your login needs a password or
  a DUO prompt, the run will wait with no output.
- The 3D plots draw a random subsample of the cloud. Use the slider to trade
  detail for speed.
- `pipeline/ui/pipeline_io.py` can be run on its own for a quick text
  report, without starting the server:

  ```bash
  python pipeline/ui/pipeline_io.py
  ```
