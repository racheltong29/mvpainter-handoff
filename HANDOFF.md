# MVPainter handoff

This bundle hands off the MVPainter texturing baseline to a new contributor
(undergrad mentee). It contains:

| File | Purpose |
|---|---|
| `HANDOFF.md` (this file) | Overview, what works, what's open, environment setup |
| `UPSTREAM_PATCHES.md` | Every modification we made to upstream MVPainter, with rationale |
| `MERGE_BACK.md` | How to integrate your work back into the main `3D-normal-cond-texture-gen` repo |
| `mvpainter_clone.patch` | Unified diff (`git apply`-able) of all upstream patches |
| `render_mvpainter_views.py` | Standalone Blender script that renders MVPainter's 6 ortho conditioning views (color / normal / depth) |
| `overcast_soil_puresky_4k.exr` | HDRI used by the render script (matches upstream) |
| `eval_assets_e2e/` | 50 production-canonical test assets, each with textured GLB + textureless GLB + 8 GT modality renders |

## 1. What MVPainter is

Multi-View Painter — a texture-generation baseline for 3D meshes. Given an
untextured mesh and a single 2D reference image, it produces a textured GLB
with PBR materials (basecolor + metallic + roughness). Internally:

1. **infer_multiview.py** — renders 6 orthographic views (color + geometric
   normal + depth) of the untextured mesh, then runs a controlnet'd
   multi-view diffusion model conditioned on the reference image to paint
   those 6 views.
2. **infer_pbr.py** (only when `--use_pbr` is passed) — runs an IDArb-derived
   diffusion model that decomposes each painted view into albedo / normal /
   metallic / roughness maps.
3. **infer_paint.py** — back-projects all 6 views per modality onto the mesh
   via a custom CUDA rasterizer + differentiable renderer, bakes them into
   UV-space textures (one per modality), and exports a glTF 2.0 PBR GLB.

Upstream repo: <https://github.com/amap-cvlab/MV-Painter>
Pinned commit we tested against: `f714682db5e6f78e8d1c7c0a43d1525638dda5a6`
(branch `main`).

## 2. Problem statement (what to work on)

We integrated MVPainter as a baseline for our texture-normal eval pipeline.
Most assets work end-to-end with `--use_pbr`, but **10 of 50 assets in our
sharded eval failed** with the same root cause: **MVPainter's bake stage
exceeds the 600 s per-asset timeout** in our sharded driver.

The numbers landed in two phases:
- **First-pass run** (`logs/run_3d_method_sharded/mv.run1/`): 12 / 50 succeeded.
  The 38 that failed went into `_triage/mv_retry_uids.txt`.
- **Retry run** (`logs/run_3d_method_sharded/mv/`): 28 / 38 of the retries succeeded.
- **Total on the 50-asset eval set:** 40 / 50 succeeded, 10 still failing —
  those 10 are listed in `_failed_uids.txt` and are your inheritance.

The failures are not in MVPainter's neural inference (multiview + IDArb work
fine), they are in `infer_paint.py`'s bake pipeline:

- `bake_pipeline.py` runs three sequential per-modality bakes (basecolor →
  metallic → roughness), each spawning a fresh CUDA rasterizer + Blender
  export. On heavy or highly curved meshes the wall clock for one asset
  exceeds 10 minutes.
- The Crowned-Skull asset (1.2 M source faces) took ~36 min for one
  end-to-end run on an A100.

The 10 failing asset UIDs are listed in `eval_assets_e2e/_failed_uids.txt`.
Two options for the project:

| Option | Pros | Cons |
|---|---|---|
| **A.** Bump the per-asset timeout to ~2400 s and rerun | Smallest change, "just works" | Slow batch wall clock; doesn't fix the root issue |
| **B.** Optimize the bake stage | Permanent speedup for all assets | Engineering effort in mvpainter/bake_pipeline.py |

Suggested first investigation for option B: profile `bake_pipeline.py` on
the Crowned-Skull asset and find which of {graphcut texture merging, UV
inpainting (`uv_inpaint`), differentiable renderer setup, Blender OBJ export}
dominates wall clock. A quick `cProfile` should answer this.

## 3. What works (confirmed)

- End-to-end with `--use_pbr` on 40 / 50 production assets.
- End-to-end smoke test on `Meshy_AI_House_Sparrow_on_a_Co_0512001212_texture`
  and `Meshy_AI_Mushroom_Crowned_Skul_0512001127_texture` produces glTF 2.0
  GLBs with baseColorTexture + metallicRoughnessTexture (no normalTexture —
  MVPainter doesn't bake the predicted normal back into the GLB, even though
  IDArb predicts one).
- HY-Paint comparison: HY-Paint is ~24× faster than MVPainter at full
  upstream settings (90 s vs 36 min/asset) — useful context if you want to
  prioritize where to optimize.

## 4. What this bundle does NOT include

- The upstream MVPainter codebase. You will clone it yourself from
  <https://github.com/amap-cvlab/MV-Painter>, then apply the diff at
  `mvpainter_clone.patch` (see `UPSTREAM_PATCHES.md`).
- Model checkpoints from HuggingFace (shaomq/MVPainter, ~51 GB +
  lizb6626/IDArb, ~1.6 GB). `UPSTREAM_PATCHES.md` documents how to fetch
  these.
- Blender. You'll need Blender 4.5+ (LTS); older 4.2.x versions hit an
  `bpy.ops.export_scene.obj` removal we patched around. See
  `UPSTREAM_PATCHES.md` §3.

## 5. Environment setup (one-time)

We tested in a conda env named `trellis2_mvp` (Python 3.10, torch 2.5.1+cu121).
A reproducible setup script lives in the main repo at
`eval/baselines/mvpainter/setup.sh`. The key pins (extracted here so you can
build outside the main repo too):

```bash
# Python 3.10, no newer (bpy 3.6 wheel only ships for cp310/3.11)
conda create -n mvpainter python=3.10
conda activate mvpainter

# Torch: 2.5.1 + cu121 (matches our system nvcc 12.4 closely enough; pip's
# default resolver pulls torch 2.11+cu130 which DOES NOT WORK with the rest
# of the stack).
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# MVPainter requirements (with bpy==3.6.0 filtered out — see UPSTREAM_PATCHES §4)
grep -v '^bpy==' third_party/MV-Painter/MVPainter/requirements.txt > /tmp/req.txt
pip install --upgrade-strategy only-if-needed -r /tmp/req.txt
pip install --upgrade-strategy only-if-needed OpenEXR pyexr xatlas

# xformers (REQUIRED for --use_pbr, NOT optional). triton 3.1.0 needed for
# xformers' triton kernels — newer triton versions break.
pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.28.post3
pip install --no-deps 'triton==3.1.0'

# CUDA extensions (need --no-build-isolation so build env can `import torch`)
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
(cd third_party/MV-Painter/MVPainter/mvpainter/custom_rasterizer  && pip install -e . --no-build-isolation)
(cd third_party/MV-Painter/MVPainter/mvpainter/differentiable_renderer && pip install -e . --no-build-isolation)
pip install cupy-cuda12x   # needed by differentiable_renderer's voronoi_solve

# Blender 4.5 LTS (4.2.x has a glTF importer bug, see UPSTREAM_PATCHES §3)
wget https://download.blender.org/release/Blender4.5/blender-4.5.0-linux-x64.tar.xz
tar -xf blender-4.5.0-linux-x64.tar.xz -C third_party/MV-Painter/
# opencv for blender's bundled python
third_party/MV-Painter/blender-4.5.0-linux-x64/4.5/python/bin/python3.11 -m pip install opencv-python

# Checkpoints (~51 GB to /tmp/mvpainter_ckpts, then rsync into MVPainter/checkpoints/)
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=300 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='shaomq/MVPainter', local_dir='/tmp/mvpainter_ckpts', local_dir_use_symlinks=False)
"
rsync -a --no-W --timeout=300 /tmp/mvpainter_ckpts/ third_party/MV-Painter/MVPainter/checkpoints/

# IDArb stack for --use_pbr (skip the unused 3.5 GB IDArb unet; MVPainter uses
# its own fine-tuned unet_pbr from shaomq/MVPainter)
HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lizb6626/IDArb',
                  ignore_patterns=['unet/diffusion_pytorch_model.safetensors'])
"
```

After all of this, apply the patches from `mvpainter_clone.patch` (see
`UPSTREAM_PATCHES.md` §6 for the exact command).

## 6. Running it on the eval set

After setup + patching:

```bash
# Single asset
python third_party/MV-Painter/MVPainter/infer_multiview.py \
    --input_glb_dir   _docs/mvpainter_handoff/eval_assets_e2e/Meshy_AI_House_Sparrow_on_a_Co_0512001212_texture/ \
    --input_img_dir   <dir-with-color.png> \
    --output_dir      /tmp/mv_test/mv_res \
    --geo_rotation 0  --diffusion_steps 75 --seed 12

python third_party/MV-Painter/MVPainter/infer_pbr.py \
    --mv_res_dir /tmp/mv_test/mv_res/mvpainter

python third_party/MV-Painter/MVPainter/infer_paint.py \
    --mv_res_dir /tmp/mv_test/mv_res/mvpainter \
    --output_dir /tmp/mv_test/painted \
    --geo_rotation 0  --use_pbr

# Final GLB lands at /tmp/mv_test/painted/combine_pbr/<base>.glb
```

The reference image MVPainter expects is a single 2D rendering of the
target appearance. For our eval, we use `view_gt_shaded.png` (a shaded
1024×1024 RGBA render of the textured GT mesh) as the reference. The
mapping:

| Input slot | What we pass |
|---|---|
| `--input_glb_dir/<asset>.glb` | `_input_mesh.glb` (textureless, from our bundle) |
| `--input_img_dir/<asset>.png` | `view_gt_shaded.png` (reference rendering) |

## 7. Inspecting what MVPainter sees internally

To debug or understand the conditioning, use the render script to reproduce
MVPainter's 6-ortho conditioning views *outside* the full pipeline:

```bash
third_party/MV-Painter/blender-4.5.0-linux-x64/blender --background -Y \
    --python _docs/mvpainter_handoff/render_mvpainter_views.py -- \
    --object_path _docs/mvpainter_handoff/eval_assets_e2e/<uid>/_input_mesh.glb \
    --output_dir  /tmp/cond_test \
    --geo_rotation 0 \
    --hdri _docs/mvpainter_handoff/overcast_soil_puresky_4k.exr
```

Output layout (matches MVPainter's `render_temp/<uid>/`):

```
/tmp/cond_test/<uid>/
    image/{000..005}.png    # 512x512 RGBA shaded views (front/right/back/left/top/bottom)
    normal/{000..005}.png   # uint16 RGB; 0.5 + 0.5*N world-space
    depth/{000..005}.exr    # OpenEXR depth, mapped to [distance-1, distance+1]
    camera/{000..005}.npy   # dict of intrinsic/extrinsic/azimuth/elevation/...
    blender.obj             # the normalized mesh, exported via Blender
```

See the docstring at the top of `render_mvpainter_views.py` for full details
on the convention.

## 8. Eval assets bundle

`eval_assets_e2e/` contains **the 50 production-canonical assets** (the UIDs
attempted in `mv.run1`). The full list is also in `_eval_uids.txt`. The
main repo has additional assets that were added after the production run
and aren't in this bundle. Per asset:

```
<uid>/
    <uid>.glb                          # textured GT mesh
    _input_mesh.glb                    # textureless mesh (textures stripped, geometry only)
    view_gt_shaded.png                 # GT shaded render (use as reference image)
    view_gt_base_color.png             # GT albedo
    view_gt_geo_normal.png             # GT geometric normal
    view_gt_tex_normal_world.png       # GT texture normal in world space
    view_gt_metallic.png               # GT metallic channel
    view_gt_roughness.png              # GT roughness channel
    view_gt_alpha.png                  # GT alpha (foreground mask)
    view_gt_mra.png                    # GT metallic+roughness+alpha packed
```

The 10 assets that failed in our production run (timeout) are listed in
`eval_assets_e2e/_failed_uids.txt`. Two of them are the "smoke-test pair"
that did succeed when run with no timeout —
`Meshy_AI_Mushroom_Crowned_Skul_0512001127_texture` (took ~36 min) and
`Meshy_AI_House_Sparrow_on_a_Co_0512001212_texture` (took ~50 s). The
mushroom-skull is the most relevant for studying the timeout problem since
it's the worst case.

## 9. Where to ask questions

- Main repo: <https://github.com/<your-org>/3D-normal-cond-texture-gen> (ask
  Ruihan for access). The eval baseline wrappers live under
  `eval/baselines/mvpainter/` and you'll want to read `setup.sh` +
  `run_mvpainter.py` to understand the harness we built around upstream.
- For specific failures on assets, the production log lives at
  `logs/run_3d_method_sharded/mv/gpu*.log` in the main repo (one log per
  GPU shard).
