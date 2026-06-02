# Upstream MVPainter modifications

Reading this saves you the trial-and-error of rediscovering 9 separate
gotchas in setting up MVPainter end-to-end. Each section explains a single
issue with: the **symptom**, the **root cause**, and the **fix**.

Pinned upstream commit: `f714682db5e6f78e8d1c7c0a43d1525638dda5a6` on
`origin/main` (<https://github.com/amap-cvlab/MV-Painter>).

To apply all source-file patches at once after you `git clone`:

```bash
cd MV-Painter
git apply /path/to/this/bundle/mvpainter_clone.patch
```

The patch touches 4 files (all under `MVPainter/`): `infer_multiview.py`,
`infer_paint.py`, `scripts/blender_bake.py`, `scripts/remesh_reduce_blender_script.py`.

The non-source-file fixes (env pins, Blender version, missing deps) are
documented below and reproduced in `HANDOFF.md` §5.

---

## 1. `xformers` is REQUIRED for `--use_pbr`, not optional

**Symptom:**
```
AttnProcessor2_0.__call__() got an unexpected keyword argument 'num_d'
```
from `pbr/pipelines/pipeline_idarbdiffusion.py` inside `infer_pbr.py`.

**Cause:** MVPainter's PBR pipeline binds a custom
`XFormersJointAttnProcessor` (in `pbr/models/transformer_dr2d.py`) to every
attention module via `set_use_memory_efficient_attention_xformers`. That
processor's `__call__` signature has extra kwargs (`num_d`, `num_v`,
`cd_attn`, `cv_attn`, `posemb`). If xformers is not installed, diffusers
falls back to the stock `AttnProcessor2_0`, which doesn't accept those
kwargs — so the first attention forward dies.

**Fix:** install xformers + a triton version compatible with both:

```bash
pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.28.post3
pip install --no-deps 'triton==3.1.0'
```

The triton pin is non-obvious: torch 2.5.1 ships with triton 3.1.0, but
several other deps pull in triton 3.6.0 which breaks xformers' triton
splitk_kernels (`TypeError: JITCallable._set_src() takes 1 positional
argument but 2 were given`).

**Pure-color mode (`--no_pbr`) does NOT need xformers** — the MV-diffusion
pipeline uses standard attention. So if you only care about color, you can
skip both pins.

---

## 2. `bpy==3.6.0` is not installable from PyPI for cp310

**Symptom:** `pip install -r requirements.txt` fails with no matching
distribution for `bpy==3.6.0`.

**Cause:** PyPI ships `bpy` wheels only for cp311+. Our env is cp310
because (a) Python 3.10 is the closest to upstream's `python==3.10` pin and
(b) torch 2.5.1+cu121 cp311 wheels aren't built by pytorch.org for our
combo.

**Fix:** filter the line out of `requirements.txt` before installing:

```bash
grep -v '^bpy==' requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt
```

This works because `bpy` is only used in `scripts/remesh_reduce_blender_script.py`,
and we patch that file (see §4) to fall back to trimesh when `bpy` is missing.

---

## 3. Blender 4.5 vs upstream's pin to 4.2.4

**Symptom A (with Blender 4.2.4):** `infer_multiview.py` crashes at the
Blender child stage with `TypeError: argument 'loglevel' is None` from
inside `bpy.ops.import_scene.gltf`. This is a known bug in Blender
4.2.4's glTF importer.

**Symptom B (with Blender 4.5 LTS):** `infer_paint.py`'s bake step fails
with `AttributeError: Calling operator "bpy.ops.export_scene.obj" error,
could not be found`. The OBJ exporter was renamed.

**Fix:** Use Blender 4.5.0+ (fixes Symptom A) and apply our patch to
`scripts/blender_bake.py` (fixes Symptom B). The patch:

```python
# scripts/blender_bake.py
if hasattr(bpy.ops.wm, "obj_export"):
    bpy.ops.wm.obj_export(
        filepath=fbx_path,
        export_selected_objects=False,
        export_materials=True,
    )
else:
    bpy.ops.export_scene.obj(
        filepath=fbx_path,
        use_selection=False,
        use_materials=True,
    )
```

(Both forms are in the patch — `obj_export` is taken on 4.5+, the old
`export_scene.obj` is the 4.2.x fallback.)

**Note on the symlink:** the upstream `infer_paint.py` hardcodes the
Blender binary path as `../blender-4.2.9-linux-x64/blender`. We kept that
exact directory name but pointed the inner `blender` symlink at our 4.5
install, so upstream's path still resolves. Or just patch the path in
`infer_paint.py` to point at wherever you actually unpacked Blender (see
§5).

---

## 4. `bpy` not in env → patch `remesh_reduce_blender_script.py`

**Symptom:** at the very start of `infer_paint.py`:
```
ModuleNotFoundError: No module named 'bpy'
```

**Cause:** `infer_paint.py` imports `reduce_mesh` from
`scripts/remesh_reduce_blender_script.py`, which has `import bpy` at module
top.

**Fix (in our patch):** make `bpy` optional and provide a trimesh fallback
for `reduce_mesh`. The fallback decimates with
`trimesh.simplify_quadric_decimation(face_count=target_vertices*1.8)` which
gives a reasonable approximation. For meshes already under the target vert
count, it's a copy.

---

## 5. `infer_paint.py` shells out via `os.system("python ...")`

**Symptom:** `infer_paint.py` runs `os.system(...)` to invoke
`bake_pipeline.py` and Blender. The shell sees the system PATH's `python`
(not the env's), which doesn't have torch.

**Cause:** `os.system(cmd)` spawns `/bin/sh -c cmd`; the env activation
doesn't propagate.

**Fix (in our patch):** at the top of `infer_paint.py`, capture
`sys.executable` and the absolute Blender path, then substitute them into
every `os.system` invocation:

```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON = sys.executable
_BLENDER = os.path.join(_THIS_DIR, '..', 'blender-4.2.9-linux-x64', 'blender')
# ...
cmd = f"cd {_THIS_DIR}/mvpainter/ && {_PYTHON} bake_pipeline.py ..."
```

**Important:** if you unpack Blender elsewhere, change `_BLENDER` in this
patch. The simplest fix is to leave the `blender-4.2.9-linux-x64` directory
name and put your 4.5 binary inside it (or symlink).

---

## 6. `blender_bake.py` doesn't slice argv at `--`

**Symptom:** when `infer_paint.py` calls `blender --background --python
blender_bake.py -- --input_dir X --save_dir Y`, the Blender child's
argparse sees `[blender, --background, --python, blender_bake.py, --,
--input_dir, X, --save_dir, Y]` and errors on `--background`.

**Cause:** Blender does NOT filter `sys.argv` at `--` for the user's
Python script.

**Fix (in our patch):**

```python
# scripts/blender_bake.py — at the top
if "--" in sys.argv:
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
```

Already done in the patch.

---

## 7. `infer_multiview.py`: depth-EXR channel `'V'` not present in Blender 4.5

**Symptom:** `infer_multiview.py:depth_exr_to_png` errors with channel
`'V'` not found in the EXR header.

**Cause:** Older Blender wrote a single-channel "V" (value) EXR for depth.
Blender 4.5 writes RGBA by default with the depth replicated across R/G/B.

**Fix (in our patch):** fall back through `'R'`, `'Y'`, `'Z'` if `'V'`
isn't present.

---

## 8. `infer_multiview.py`: subprocess cwd / OOM

**Symptom A:** the Blender subprocess for ortho rendering can't find
`scripts/blender_render_ortho.py` — error from inside Blender about a
missing script.

**Cause:** `infer_multiview.py` calls `subprocess.run(cmds)` with a
relative script path. Whether it works depends on the parent's cwd.

**Fix (in our patch):** resolve to absolute paths and pass
`cwd=_MVP_DIR` to the subprocess.

**Symptom B (separate):** `cuda out of memory` at pipeline build, even on
80 GB GPUs, when other workloads share the GPU.

**Cause:** the UNet checkpoint is loaded directly to CUDA in upstream
(`torch.load(..., map_location='cuda')`), peaking memory before
`pipeline.to('cuda')` can run.

**Fix (in our patch):**
1. Load the ckpt to CPU first (`map_location='cpu'`).
2. After `pipeline.unet.load_state_dict(new_ckpt)`, `del` the temp dicts
   and call `gc.collect()` + `torch.cuda.empty_cache()`.
3. Use `pipeline.enable_model_cpu_offload()` (lighter than
   `enable_sequential_cpu_offload`, which trips a meta-tensor error on the
   freshly loaded UNet).

The patch includes a fallback that drops to full-GPU
`pipeline.to('cuda')` if `enable_model_cpu_offload` errors out.

---

## 9. `differentiable_renderer` needs `cupy`

**Symptom:** at bake time, `voronoi_solve` in `differentiable_renderer`
errors with `ModuleNotFoundError: No module named 'cupy'`.

**Cause:** cupy isn't listed in `requirements.txt`.

**Fix (not in the patch — env-level):**

```bash
pip install cupy-cuda12x
```

(Or `conda install -c conda-forge cupy` per upstream INSTALL.md, but pip is
faster.)

---

## Summary table

| Issue | Where | In patch? |
|---|---|---|
| 1. xformers required for --use_pbr | env-level | No (env install) |
| 2. bpy not on PyPI for cp310 | env-level | No (filter requirements.txt) |
| 3a. Blender 4.2.4 glTF importer bug | env-level | No (use 4.5) |
| 3b. Blender 4.5 export_scene.obj removed | `scripts/blender_bake.py` | **Yes** |
| 4. bpy module missing | `scripts/remesh_reduce_blender_script.py` | **Yes** |
| 5. Wrong python in os.system | `infer_paint.py` | **Yes** |
| 6. Argv `--` slicing | `scripts/blender_bake.py` | **Yes** |
| 7. Depth EXR channel | `infer_multiview.py` | **Yes** |
| 8a. Subprocess cwd | `infer_multiview.py` | **Yes** |
| 8b. UNet OOM at load | `infer_multiview.py` | **Yes** |
| 9. Missing cupy | env-level | No (pip install) |

If you hit anything not in this list, check
`HANDOFF.md` §5 (env setup) first, then look at
`eval/baselines/mvpainter/setup.sh` in the main repo for the exact pinned
versions we use in CI.
