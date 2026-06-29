Session date: 2026-06-23. Files touched: `MVPainter/scripts/remesh_reduce_blender_script.py`,
`MVPainter/infer_paint.py`, `MVPainter/scripts/run_reduce_mesh.py` (new),
`MVPainter/run_full_pipeline.sh` (new), and the `mvpainter` conda env's
`activate.d`/`deactivate.d` hooks (new).

Context: the handoff's reported problem was 10/50 eval assets failing on a
600s bake timeout (see `HANDOFF.md` §2). While investigating, a **separate,
previously-undiagnosed bug** was found: output 3D files were fragmented into
grey, disconnected triangles for large meshes. This doc covers everything
found while tracking that down, in the same Symptom / Root cause / Fix format
as `UPSTREAM_PATCHES.md`.

---

## 1. `bpy` cannot be imported in Python env

File: `MVPainter/infer_paint.py` (caller of `reduce_mesh`)

**Symptom:** `ModuleNotFoundError: No module named 'bpy'` when `infer_paint.py`
calls `reduce_mesh()` directly in-process.

**Cause:** `infer_paint.py` runs under the `mvpainter` conda env (Python
3.10). PyPI's `bpy` package has never shipped a `cp310` wheel for **any**
version. This was confirmed against PyPI's full release JSON and when I ran `pip install bpy==3.6.0 --dry-run`, it reported that it `Could not find a version that satisfies
the requirement bpy==3.6.0 (from versions: none)`. Every `bpy` release is
`cp311`-only (4.2.0–5.0.1) or `cp313`-only (5.1.0+).

> **Note:** `HANDOFF.md` §5 says *"bpy 3.6 wheel only ships for cp310/3.11"* —
> this is **incorrect** and should be fixed in that doc. The 3.10 pin in
> `HANDOFF.md` is real and necessary, but not because of `bpy` — see the
> Python 3.11 compatibility test below.

A trimesh-based fallback existed for this as part of the previous UPSTREAM_PATCHES but itself had errors with decimation that reduced it to broken output Blender files.
**Fix:** `reduce_mesh` is no longer called in-process. `infer_paint.py` now
invokes it via subprocess, running the real Blender binary:
```bash
blender --background --python scripts/run_reduce_mesh.py -- \
    --input_obj <obj> --output_obj <low_obj> --target_vertices 30000
```
(`scripts/run_reduce_mesh.py` is a new small CLI wrapper that imports and
calls `reduce_mesh()`.) This mirrors the pattern already used elsewhere in
the codebase for `blender_bake.py` and `blender_pbr_glb.py`.

I also tried a separate Python 3.11 conda env (`bpy311`) with `pip install
bpy` as an alternative to launching the full Blender app 
This also works
(confirmed `bpy` 5.0.1 imports and runs `reduce_mesh` correctly), but we ended
up using the Blender-binary approach instead since it requires no extra env
and matches the existing codebase convention. The `bpy311` env is left in
place in case it's useful for quick scripting.

---

## 3. Blender 4.0+ removed the OBJ import/export operators we were using

File: `MVPainter/scripts/remesh_reduce_blender_script.py` both
`reduce_mesh()` and `remesh_and_replace()`

**Symptom** (first time `reduce_mesh()` actually ran against real Blender 4.5,
via the Blender binary):
```
File ".../remesh_reduce_blender_script.py", line 48, in reduce_mesh
    bpy.ops.import_scene.obj(filepath=input_obj_path)
AttributeError: Calling operator "bpy.ops.import_scene.obj" error, could not be found
```

**Root cause:** `bpy.ops.import_scene.obj` and `bpy.ops.export_scene.obj` are
the old OBJ import/export operators from pre-4.0 Blender. Blender 4.0+ removed
them and replaced them with `bpy.ops.wm.obj_import` / `bpy.ops.wm.obj_export`.
This is a **known, already-documented issue** in this codebase —
`UPSTREAM_PATCHES.md` §3 describes hitting this exact `AttributeError` on
`export_scene.obj` and patching `scripts/blender_bake.py` for it. The same fix
was applied to `scripts/blender_pbr_glb.py` and `scripts/blender_render_ortho.py`
at some point too. `remesh_reduce_blender_script.py` never got this fix,
because (per bugs 1 and 2 above) it had never actually run successfully
against real Blender before — it always either crashed on the `bpy` import or
hit the dead assert.

**Fix:** replaced `bpy.ops.import_scene.obj(...)` → `bpy.ops.wm.obj_import(...)`
and `bpy.ops.export_scene.obj(...)` → `bpy.ops.wm.obj_export(...)` at all 6
call sites (3 in `reduce_mesh`, 3 in `remesh_and_replace`). Since this script
is only ever invoked from inside actual Blender now (never imported
in-process), we did **not** add a `hasattr()` compatibility shim for the old
API like `blender_bake.py` has — we're always on Blender 4.5+ here.

---

## 4. The actual fragmentation root cause — decimation silently never ran

This is the combination of bugs 1–3. Every prior attempt to call
`reduce_mesh()` either crashed (bug 2/3) or hit the dead assert (bug 1). Since
`infer_paint.py` has:
```python
if os.path.exists(obj.replace('.obj','_low.obj')):
    continue
```
...any `_low.obj` file already sitting on disk from an earlier, broken attempt
got reused forever, never regenerated.

We compared a stale cached `blender_low.obj` against its source `blender.obj`
for two assets (`86df7065d407ef41d923469c1bcb95e6` and the mushroom-skull
asset) and found:

- **Face count: identical** between `blender.obj` and `blender_low.obj` (e.g.
  497607 == 497607) → decimation never actually reduced anything.
- **Vertex count: ~3x** the vertices of `blender.obj` (e.g. 248075 → 1492054,
  almost exactly 3x the face count) → the mesh was exported "exploded": every
  triangle got its own private, unshared vertices instead of sharing vertices
  with its neighbors.

An exploded/unwelded mesh like this breaks UV unwrapping and texture
projection downstream (every triangle becomes its own disconnected island),
which is what produced the originally-reported symptom: grey, fragmented,
far-apart triangular faces in the final baked output, specifically on larger
meshes (~80k+ faces) where decimation should have been doing real work.

**Fix:** covered by bugs 1–3 above. With those fixed, `reduce_mesh` actually
welds (`remove_doubles`) and decimates the mesh via Blender's `DECIMATE`
modifier before export. Verified on multiple assets that output vertex count
now tracks the expected ~F/2 ratio for a properly connected triangle mesh, not
~3x face count.

---

## 5. `LD_LIBRARY_PATH` missing torch's lib dir → `custom_rasterizer` ImportError

**Symptom:**
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```
...when `bake_pipeline.py` tries `import custom_rasterizer`.

**Root cause:** `custom_rasterizer_kernel` is a compiled CUDA extension built
against PyTorch, so it needs to link against one of PyTorch's own shared
libraries, `libc10.so`, at runtime. It wasn't built with an rpath pointing at
PyTorch's lib folder, and this shell's `LD_LIBRARY_PATH` only included the
CUDA toolkit path (`/usr/local/cuda-13.1/lib64`), not
`.../envs/mvpainter/lib/python3.10/site-packages/torch/lib/` where `libc10.so`
actually lives.

**Fix:** added a conda env hook scoped to `mvpainter` so this is automatic on
`conda activate mvpainter`:
- `envs/mvpainter/etc/conda/activate.d/env_vars.sh` — prepends torch's lib dir
  onto `LD_LIBRARY_PATH`, saving the old value first.
- `envs/mvpainter/etc/conda/deactivate.d/env_vars.sh` — restores the saved old
  value on `conda deactivate`, so this doesn't leak into other envs
  (`trellis2`, `syncmvd`, etc.).

Verified: `LD_LIBRARY_PATH` correctly includes `torch/lib` right after
activate, `custom_rasterizer` imports cleanly, and the value correctly reverts
after deactivate.

---

## 6. Decimate loop can spin forever on topologies that hit a structural floor

File: `MVPainter/scripts/remesh_reduce_blender_script.py`, `reduce_mesh()`

**Symptom:** while running the full batch on
`Meshy_AI_Blonde_hair_0511224224_texture`, the decimate loop hit iteration
4739 (!), printing the exact same verts/faces count (37795 verts / 81923
faces) on every iteration from ~4735 onward. It had been spinning for ~85
CPU-minutes with zero progress when caught.

**Root cause:** the loop was:
```python
while current_poly_count > target_vertices:
    # apply one DECIMATE modifier pass
    current_poly_count = len(obj.data.vertices)
```
For most meshes, each pass reduces the count further until it drops below
`target_vertices`. But some topologies (this asset has lots of thin,
disconnected hair-strand geometry) hit a structural floor where Blender's
`DECIMATE` modifier simply can't collapse any further without destroying the
mesh — `current_poly_count` stops decreasing, but since it's still above
`target_vertices`, the `while` condition stays true forever and the loop
reapplies a no-op modifier indefinitely.

**Fix:** track the previous iteration's vertex count and break out of the loop
early if the new count is `>=` the previous count (i.e. no progress was made),
logging that decimation stalled:
```python
new_poly_count = len(obj.data.vertices)
...
if new_poly_count >= current_poly_count:
    print(f"... decimate stalled (no progress past {new_poly_count} "
          f"verts, target was {target_vertices}) - stopping early")
    current_poly_count = new_poly_count
    break
current_poly_count = new_poly_count
```
This is **not** really a "`target_vertices` is too low" problem in general —
it's that certain mesh topologies (thin/disconnected strand-like geometry)
have a hard floor below which this decimation approach cannot go, independent
of what `target_vertices` is set to.

Verified: re-running on the same asset now stops at iteration 13 (still
plateaued at 37795 verts, same structural floor) instead of looping forever —
finishes in seconds instead of hanging indefinitely.

---

## Things investigated that turned out not to be bugs

- **Final GLB texture looked lighter than the input reference image.** Traced
  this back stage-by-stage: the raw painted multiview output (`result_1.png`,
  straight from `infer_multiview.py`'s diffusion model, before *any* of our
  baking/projection code runs) was already lighter than the reference. So
  this is the model's own behavior (likely IDArb's albedo decomposition
  flattening lighting, or the diffusion model's own output characteristics) —
  not introduced by anything in the decimation/baking pipeline.

- **Re-running the same asset with the same `--seed` produced visibly
  different painted output between two separate runs.** This is expected:
  xformers' memory-efficient attention (required for `--use_pbr` per
  `HANDOFF.md`) does not guarantee deterministic floating-point operation
  order across runs, so a fixed seed alone does not guarantee reproducible
  output. Not a regression from anything changed here.

- **A relative-path bug in our own new one-shot wrapper script**
  (`run_full_pipeline.sh`) caused some Blender subprocess steps to silently
  write output to the wrong directory when `--output_dir` was relative
  (because those subprocess calls `cd` into a different directory before
  resolving the path). Fixed by always resolving to absolute paths before
  passing them to any subprocess. This was a bug in the new convenience
  script we wrote this session, not in the original MVPainter code.

- **`infer_multiview.py` pairs glb/image files by matching basename**
  (`X.glb` ↔ `X.png`). Our one-shot script was copying `eval_assets_e2e`'s
  files (`_input_mesh.glb` + `view_gt_shaded.png`, different basenames) into
  temp dirs without renaming, so the pairing silently matched nothing and the
  multiview stage did nothing (no error). Fixed by copying both to a shared
  basename (`asset.glb` / `asset.png`) in the wrapper script. Also a bug in
  our new script, not original MVPainter code.

---

## Python 3.11 compatibility test (re: "is the 3.10 pin really necessary")

`HANDOFF.md`'s stated reason for pinning Python 3.10 ("bpy 3.6 wheel only
ships for cp310/3.11") is factually wrong per bug 2 above. So we tested
whether the **rest** of the pinned stack (`torch==2.5.1+cu121`,
`xformers==0.0.28.post3`, `triton==3.1.0`, the `custom_rasterizer` and
`differentiable_renderer` CUDA extensions) actually requires 3.10, or whether
that was just an unverified assumption.

**Result:** built a fresh conda env (Python 3.11) and successfully installed
and imported, together in one process: `bpy` 5.0.1, `torch` 2.5.1+cu121 (CUDA
available, confirmed with a real `.cuda()` tensor op), `xformers` 0.0.28.post3,
`triton` 3.1.0 (pulled in automatically by torch), `cupy`, and both
custom-compiled extensions (`custom_rasterizer`, `differentiable_renderer`)
built from source against this env with no errors. So the whole stack **does**
work on Python 3.11, including `bpy` directly in-process — the 3.10 pin in
`HANDOFF.md` is not actually required for any of this.

One real gotcha hit along the way: the latest `cupy-cuda12x` (14.1.1) requires
`numpy>=2.0`, which conflicts with `scipy==1.13.0`'s pin
(`numpy<2.3,>=1.22.4`) and the documented `numpy==1.26.3`. This is unrelated
to the Python 3.10-vs-3.11 question (package versions have simply drifted
since `HANDOFF.md` was written; it would hit the same wall on 3.10 today too)
— pin `cupy-cuda12x==13.3.0` instead, which accepts `numpy<2.3,>=1.22`.
