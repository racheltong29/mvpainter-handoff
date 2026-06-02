# Merging your work back into the main repo

When you're ready to integrate your MVPainter improvements into the main
`3D-normal-cond-texture-gen` repo, this doc tells you the layout, contracts,
and PR conventions used there.

## Repo layout you'll target

```
3D-normal-cond-texture-gen/
├── eval/baselines/
│   ├── README.md                       # baseline overview
│   ├── _common.py                      # shared helpers (strip_textures_glb, validate, ...)
│   ├── sentinel_uids.txt               # 3-asset smoke list
│   ├── mvpainter/
│   │   ├── env.yml, setup.sh           # env spec
│   │   ├── run_mvpainter.py            # the wrapper you'll touch
│   │   └── postprocess.py              # GLB post-processing (mostly noop)
│   ├── hunyuan3d_paint/                # parallel structure for HY-Paint baseline
│   └── dreammat/                       # parallel structure for DreamMat
└── third_party/
    └── MV-Painter/MVPainter/           # gitignored clone, you apply patches into this
```

## The wrapper contract

`eval/baselines/mvpainter/run_mvpainter.py` is the entry point our eval
harness calls. Its job is:

1. Read each `<eval_dir>/<asset>/{gt.glb, color.png}` (and
   `tex_normal_ws.png` — gates `collect_asset_dirs` even though MVPainter
   doesn't use it).
2. Stage them into a temp dir under `/tmp/mvpainter_*/` with the layout
   `glbs/<base>.glb + imgs/<base>.png` that upstream `infer_multiview.py`
   wants.
3. Call `infer_multiview.py`, then `infer_pbr.py` (if `--use_pbr`), then
   `infer_paint.py`.
4. Copy the final GLB to `<eval_dir>/<asset>/textured_mvpainter.glb`.
5. Validate (size cap < 50 MB, has baseColorTexture, etc.).

If your changes only touch the upstream clone (in `third_party/MV-Painter/`),
you don't need to modify `run_mvpainter.py` at all — just make sure the
`combine_pbr/<base>.glb` output still lands where the wrapper expects it.

## Output GLB contract

The eval harness expects per-asset output:

| Property | Value |
|---|---|
| Path | `<eval_dir>/<asset>/textured_mvpainter.glb` |
| Format | glTF 2.0, single mesh (multi-primitive OK) |
| baseColorTexture | RGB, required |
| metallicRoughnessTexture | G=rough, B=metal, optional but improves the metric |
| normalTexture | tangent-space, optional |
| File size | < 50 MB |
| Pose | Aligned to `gt.glb` so the turntable cameras hit the same content |

`validate_output_glb()` in `eval/baselines/_common.py` parses + sanity-checks
these. Look at how `postprocess_mv_glb` is currently a near-noop — if your
changes break any of the above, this is where you'd add a fixer pass.

## Workflow we use

1. **Pick a branch off `main`.** Use a descriptive name like
   `<your-andrew-id>-mvpainter-bake-speedup`.

2. **Develop in your standalone bundle first.** Keep iterating without
   touching the main repo. Faster cycles since you don't need to merge or
   rebase.

3. **When you have a working improvement, integrate by:**
   - Copy your modified upstream files into the clone under
     `third_party/MV-Painter/MVPainter/`. (The `third_party/` dir is
     gitignored, so changes here don't become commits — they're recorded
     via a regenerated `_docs/mvpainter_handoff/mvpainter_clone.patch`.)
   - Run `git -C third_party/MV-Painter diff >
     _docs/mvpainter_handoff/mvpainter_clone.patch` to update the
     handoff patch.
   - If you changed the wrapper (`eval/baselines/mvpainter/run_mvpainter.py`),
     re-run the sentinel smoke test before committing:
     ```bash
     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
     LD_LIBRARY_PATH="/root/miniconda3/envs/trellis2_mvp/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH" \
       /root/miniconda3/envs/trellis2_mvp/bin/python eval/baselines/mvpainter/run_mvpainter.py \
         --eval_dir eval_pipeline/ --sentinels --gpu 1 \
         --geo_rotation 0 --diffusion_steps 30 --use_pbr --overwrite
     ```

4. **Push and open a PR.** Title format: `mvpainter: <one-line summary>`.
   Body should include:
   - **Summary:** 1–3 bullets on what changed and why.
   - **Test plan:** what you ran, what you saw. Include before/after timing
     numbers if your change is a perf improvement.
   - **Limitations:** anything left unfixed.

5. **CI runs the sentinel smoke test automatically.** If it passes, ask
   Ruihan for review.

## Things to be careful with

- **Don't `git add` anything under `third_party/`.** It's gitignored, but
  if you ever force-add a file there it pollutes the repo. The
  `mvpainter_clone.patch` *is* the source of truth for upstream
  modifications.
- **`fugitive/eval_pipeline/` is large (hundreds of GB).** Don't `cp -r`
  it. Use `find` / symlinks.
- **Conda envs:** the main repo has *separate* envs per baseline
  (`trellis2_mvp`, `trellis2_hyp`, `trellis2_dm`). They conflict with each
  other (especially on torch / cuda versions). Don't try to install
  MVPainter requirements into the wrong env.
- **The `_input_mesh.glb` files in `eval_assets_e2e/` are part of the
  handoff bundle.** If you generate similar files in the main repo's
  `datasets/eval_assets_e2e/`, drop them next to the existing layout —
  the wrapper's `find_input_mesh` will pick them up automatically (it
  prefers `_input_mesh.glb` over `gt.glb`).

## Quick reference

| You want to... | Look at |
|---|---|
| Add a new flag to the wrapper | `run_mvpainter.py:main()` |
| Change the output GLB layout | `postprocess.py` + `validate_output_glb` in `_common.py` |
| Profile MVPainter's bake | `third_party/MV-Painter/MVPainter/mvpainter/bake_pipeline.py` |
| Change which views MVPainter renders | `third_party/MV-Painter/MVPainter/scripts/blender_render_ortho.py` |
| Update the IDArb PBR step | `third_party/MV-Painter/MVPainter/infer_pbr.py` |
| Re-bundle the handoff | `_docs/mvpainter_handoff/` + `_docs/mvpainter_handoff/_strip_textures.py` |

Welcome aboard — happy texturing.
