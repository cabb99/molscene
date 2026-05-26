# CaMKII–actin holoenzyme transplant

A worked example that uses the MolScene `Transformation` / `Matching` /
`morph_segment` machinery to rebuild a CaMKII dodecamer with its 12 kinase
domains repositioned along two actin filaments, taking the kinase–actin
binding pose from Qian's model and the holoenzyme architecture from the
"merged" template.

## Inputs

| File | What it is |
| ---- | ---------- |
| `merged_CaMKII.pdb` | Original template: 12 full CaMKII subunits (chains A–L) plus two 13-monomer actin filaments (chains M–l). |
| `CaMKII_qian_template.pdb` | Qian's single-chain model: one CaMKII kinase domain bound to two consecutive actin monomers, with the dodecameric hub and the rest of the actin filaments fused into the same chain A. |

## Outputs

Produced by `script.py`:

| File | When | What it is |
| ---- | ---- | ---------- |
| `camkii_candidates.pdb` | `--save-candidates camkii_candidates` | The 12 source CaMKII chains (A–L, kinase + linker + hub) **plus** every candidate kinase pose, each on its own one-char chain (`0`–`9`, `m`–`x`). Open in PyMOL / VMD / ChimeraX and pick which candidate each source kinase should map onto. |
| `camkii_candidates.txt` | `--save-candidates camkii_candidates` | A sortable table: candidate id, chain label in the PDB, filament index, actin-monomer pair, fit RMSD, distance to hub centroid. Plus the centroid of each source kinase so you can eyeball which is closest to which candidate. |
| `camkii_on_actin.pdb` | default / `--assignment` | Final structure: hub ring and actin filaments untouched, the 12 kinase domains repositioned per the assignment, linkers smoothly ScLERP-morphed between the unchanged hub end and the moved kinase end. |
| `camkii_on_actin_movie.pdb` | `--animate N` | Multi-model PDB of N frames showing the morph from rest (α = 0) to the final pose (α = 1). Each frame is a standard `MODEL`/`ENDMDL` block. |

## Three usage modes

### 1. Auto-assignment (default)

```bash
python script.py
```

Ranks all 22 candidate kinase poses by distance to the CaMKII hub centroid,
keeps the top 12, then uses the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) to match each merged chain A–L to
its nearest candidate. Writes `camkii_on_actin.pdb`.

### 2. Manual inspection + manual assignment

Step 1 — save the candidate poses for visual inspection:

```bash
python script.py --save-candidates camkii_candidates
```

Open `camkii_candidates.pdb` in your viewer. The source kinases (chains
A–L) show you where each CaMKII subunit currently sits and where the hub
ring is anchored; the candidate poses (chains `0`–`9`, `m`–`x`) show every
viable kinase docking position along the two actin filaments. Cross-reference
with `camkii_candidates.txt` to read the candidate id, fit RMSD and hub
distance for each.

Step 2 — write a text file mapping each source chain to a candidate id:

```
# camkii_assignment.txt — one line per source chain.
# Format: "source_chain candidate_id". Lines starting with '#' ignored.
A   0
B   3
C  17
...
L  11
```

The file must list all twelve source chains; each candidate id must be
unique. Step 3 — apply it:

```bash
python script.py --assignment camkii_assignment.txt
```

### 3. Filament-rotation optimisation

Once the kinases are in place, the linkers between hub and kinase
generally have unequal lengths — some stretched, some bunched. Treating
each linker as a spring, we can rotate each actin filament around its
own helical axis to balance the spring tensions. The kinases swing
around with the filament (rotations around the helical axis are an
internal degree of freedom of the bound system), and the linker spans
adjust accordingly.

Add `--optimize-filaments`:

```bash
python script.py --assignment camkii_assignment.txt --optimize-filaments
```

For each of the two filaments, the script:

1. Computes the helical axis from the filament's atom coordinates
   (centroid → point on axis; first principal component → direction —
   equivalent to fitting a z-aligned synthetic filament and reading the
   rotation off the Kabsch result).
2. Coarsely scans rotation angle 0°–360° (every `--scan-step` degrees,
   default 1°) and finds the angle that minimises the sum of squared
   linker spans for the chains bound to that filament. The "linker
   span" is the distance between the CA of the first linker residue
   (`LINKER_RANGE[0]`, anchored to the kinase end) and the CA of the
   last linker residue (`LINKER_RANGE[1]`, anchored to the hub end).
3. Refines the coarse winner with Brent's method.
4. Prints the spans per chain before and after, so you can see the
   tightening.

The sum-of-squares objective tends to equalise the spans (by penalising
outliers) more than it minimises the absolute mean; expect the standard
deviation of the per-chain spans to drop more than the mean. On the
provided v3 assignment, std went from 22.3 Å → 15.6 Å.

`--optimize-filaments` composes cleanly with `--animate N`; in the
movie, both the actin filaments and the kinases gradually rotate
together, with the linkers ScLERP-morphing between them.

### 4. Symmetry enforcement

Two independent flags re-pose the assembly to satisfy idealised symmetries.
Each can be used alone or combined with the other (and with `--animate`).
A four-line geometry printout of the filament pair runs on every
invocation, before and after, so you can see how the descriptors change.

**`--enforce-symmetry {parallel,antiparallel}`** — reposes filament 1 so
it is the C2-image of filament 0 about the CaMKII hub COM:

* `parallel` — C2 axis = filament 0's helical-axis direction. Filament 1
  ends up pointing the same way as filament 0, on the opposite side of
  the CaMKII; the radial reference is rotated by 180°.
  → output `angle_deg ≈ 0°`, `axial_offset ≈ 0`, `twist ≈ 180°`.
* `antiparallel` — C2 axis = `d̂₀ × n̂_⊥` (perpendicular to both the
  helical axis and the perp-vector to the CaMKII COM). Filament 1 ends
  up pointing the opposite way and mirrored across the CaMKII COM.
  → output `angle_deg ≈ 180°`.

This is a frame-alignment transform (not a literal 180° rotation of
filament 1's current pose) — it physically moves filament 1 to the
position the C2 symmetry says it should occupy. The kinases bound to
filament 1 move with it; the linker morph absorbs the shift.

Compatible with `--optimize-filaments`: if both are set, filament 0 is
optimised and filament 1 is symmetrised.

**`--enforce-camkii-symmetry {pointed,flat}`** — repositions the CaMKII
hub: translates so the hub COM sits at the midpoint of the closest-
approach segment between the two filament axes, then orients the
dodecameric hexagon so

* `pointed` — vertex AB direction ⊥ to filament 0's helical axis (a
  vertex points perpendicular toward each filament).
* `flat` — the HK ↔ DE diameter ‖ to filament 0's helical axis (a hexagon
  diameter is along the filament length; equivalently, an edge faces
  perpendicular to the filaments).

Pointed and flat differ by a 30° rotation of the hexagon about its stack
axis. Only the **hub residues** (`HUB_RANGE`) of all 12 CaMKII chains
are transformed; the kinases stay bound to their candidate positions and
the linker morph re-runs with `t_start = T_cam` so the hub-end of the
linker tracks the moved hub.

**The four canonical configurations** the user typically wants:

```bash
python script.py --assignment ASSIGN --enforce-symmetry parallel     --enforce-camkii-symmetry pointed
python script.py --assignment ASSIGN --enforce-symmetry parallel     --enforce-camkii-symmetry flat
python script.py --assignment ASSIGN --enforce-symmetry antiparallel --enforce-camkii-symmetry pointed
python script.py --assignment ASSIGN --enforce-symmetry antiparallel --enforce-camkii-symmetry flat
```

Each writes its own `--output` PDB; add `--animate N` to also dump the
corresponding morph movie.

### 5. Animation

Add `--animate N` to either mode to also write an N-frame multi-model PDB:

```bash
python script.py --assignment camkii_assignment.txt --animate 20
python script.py --assignment camkii_assignment.txt --optimize-filaments --animate 20
```

This produces `camkii_on_actin_movie.pdb` (≈10 MB per frame, so ~200 MB at
N=20). Open it as a trajectory:

| Viewer | Open as |
| ------ | ------- |
| PyMOL | `load camkii_on_actin_movie.pdb, mov, multiplex=0` |
| VMD | `mol new camkii_on_actin_movie.pdb` (auto-detected) |
| ChimeraX | `open camkii_on_actin_movie.pdb` (auto-detected) |

Without `--optimize-filaments`, hub and actins stay fixed across frames;
only the 12 kinase domains and their linkers move. With
`--optimize-filaments`, the two actin filaments also rotate around their
own helical axes.

## What the pipeline does

1. **Find the kinase-binding actins in Qian.** Pairwise min-atom distance
   between the kinase (`chain A and resid 1 to 290`) and each of the five
   actin monomers; the two closest are the KABT (Kinase-Actin Binding
   Template) actins.
2. **Set up the merged filaments.** KMeans on per-chain centroids splits the
   26 actin chains into two filaments; each is ordered along its own
   principal axis via SVD.
3. **Slide KABT with stride 2.** For every same-strand monomer pair
   `(filament[i], filament[i+2])`, Kabsch-fit KABT_actins onto the pair
   (residue correspondence via `SequenceMatching`), then apply that rigid
   `Transformation` to KABT_kinase. Gives 11 candidates per filament, 22
   total.
4. **Assign.** Either Hungarian on hub-distance ranking, or your manual file.
5. **(Optional) Per-filament rotation.** Two mutually-compatible options:
   * `--optimize-filaments`: rotate each filament about its helical axis
     to minimise the sum of squared linker spans (coarse 1° scan +
     Brent refinement).
   * `--enforce-symmetry {parallel,antiparallel}`: replace filament 1's
     pose with the C2-image of filament 0 (parallel or antiparallel,
     respectively).
   When both are set, filament 0 is optimised and filament 1 is symmetrised.
6. **(Optional) `--enforce-camkii-symmetry {pointed,flat}`.** Build a
   single rigid `Transformation` that translates the hub COM to the
   filament midpoint and reorients the hexagonal dodecamer (vertex AB
   perpendicular to the helical axis for `pointed`, HK↔DE diameter
   parallel for `flat`). Applied to the hub residues only.
7. **Move kinases + morph linkers.** For each assigned source chain,
   `compute_transformation()` puts the source kinase onto the candidate,
   `transform()` applies that to the kinase residues, and
   `morph_segment(method='sclerp')` ScLERP-interpolates the linker.
   The morph's `t_start` is identity (or the CaMKII symmetry transform
   if `--enforce-camkii-symmetry` was given), the `t_end` is the
   composed kinase transform.
8. **Write** the final PDB (and the movie if `--animate` was given).
   The four-line filament-pair geometry printout runs before and after
   so you can see how the assembly's geometry changed.

## Notes / caveats

- Residue boundaries are hard-coded near the top of `script.py`
  (`KINASE_RANGE = (1, 290)`, `LINKER_RANGE = (291, 400)`,
  `HUB_RANGE = (401, 542)`). Tune these for other CaMKII constructs.
- The script intentionally uses a single canonical actin-pair orientation
  (`KABT_CHAIN_A → fil[i]`, `KABT_CHAIN_B → fil[i+2]`) so adjacent
  overlapping stride-2 pairs produce distinguishable poses. If your
  filaments are ordered against the natural KABT polarity, swap the
  filament order in `cluster_into_filaments` or invert the KABT mapping.
- Per-residue ScLERP preserves intra-residue geometry exactly. Inter-residue
  C–N peptide bonds in the linker can stretch by a few Å — this is a
  starting model, expected to be relaxed downstream by energy minimisation
  or short MD.

## Sanity checks after a run

```bash
python - <<'PY'
import numpy as np
from molscene import Scene

before = Scene.from_pdb('merged_CaMKII.pdb')
after  = Scene.from_pdb('camkii_on_actin.pdb')

# Hub atoms should not have moved.
for c in 'ABCDEFGHIJKL':
    m = (before['chain'] == c) & (before['resid'].between(401, 542))
    delta = np.linalg.norm(
        after.loc[m, ['x','y','z']].to_numpy()
        - before.loc[m, ['x','y','z']].to_numpy(), axis=1)
    assert delta.max() < 1e-6, f'hub of chain {c} moved!'

# Actin chains should not have moved.
for c in [ch for ch in before['chain'].unique() if ch not in 'ABCDEFGHIJKL']:
    m = before['chain'] == c
    delta = np.linalg.norm(
        after.loc[m, ['x','y','z']].to_numpy()
        - before.loc[m, ['x','y','z']].to_numpy(), axis=1)
    assert delta.max() < 1e-6, f'actin chain {c} moved!'

print('OK — hub and actins are immobile.')
PY
```

## Plan: Filament Geometry Report + `--enforce-symmetry`

Add a geometry printout (runs always) and a `--enforce-symmetry` flag that couples filament 2 movement into a perfect fold filament arrangement, both for paralel actin filament, doing a 180 rotation around the COM of the CAMKII in the plane perpendicular to the helical axis, and an antiparallel actin filament, doing a 180 degree rotation around the COM of the CaMKII in the plane formed by the COM of the CaMKII and the helical axis of the filament.

---

### Phase 1 — Two new helper functions
(inserted after `optimize_filament_rotation`, before `# Pipeline stages`)

**`describe_filament_pair(merged, filaments) → geom`**
Computes the 4 numbers you asked for:
- **angle_deg** — angle between the two helical axis directions: $\arccos(\hat{d}_0 \cdot \hat{d}_1)$
- **d_perp_A** — interfilament distance ⊥ to the helical axis: $|\,(p_1-p_0) - d_\text{axial}\,\hat{d}_0\,|$
- **d_axial_A** — axial offset: $(p_1-p_0)\cdot\hat{d}_0$
- **twist_deg** — rotation of filament 1 around the helix axis relative to filament 0; measured using the first monomer centroid projected ⊥ to the axis as a radial reference: $\text{atan2}((\hat{v}_0\times\hat{v}_1)\cdot\hat{d}_0,\; \hat{v}_0\cdot\hat{v}_1)$

Returns a `geom` dict (`d0,p0,d1,p1,n_perp,v0,v1` + 4 scalars).

**`build_symmetry_transform_filament1(geom, target_distance_A=240.0) → Transformation`**
Uses a local-frame alignment:
- Current frame $R_1 = [\hat{v}_1 \mid \hat{d}_1\times\hat{v}_1 \mid \hat{d}_1]$ at $p_1$
- Target frame $R_{1,\text{tgt}} = [-\hat{v}_0 \mid \hat{d}_0\times(-\hat{v}_0) \mid \hat{d}_0]$ at $p_\text{tgt} = p_0 + D\,\hat{n}_\perp$

($\hat{n}_\perp \perp \hat{d}_0$ → axial offset = 0 automatically; $-\hat{v}_0$ → 180° twist)

$$R_\text{total} = R_{1,\text{tgt}}\, R_1^\top, \qquad t = p_\text{tgt} - R_\text{total}\,p_1$$

** similarly for the other transformation

---

### Phase 3 — Two new CLI args in `parse_args()`

- `--enforce-symmetry` (store_true)
- `--sym-distance` (float, default `240.0`, in Å)


---

### Phase 3 — `main()` changes

**3a.** Call `geom = describe_filament_pair(merged, filaments)` right after the hub centroid printout (runs on every invocation, including `--save-candidates`).

**3b.** Replace the `filament_rotations = None; if args.optimize_filaments: ...` block with a unified loop over both filaments:
- If `fi == 1` and `--enforce-symmetry`: use `build_symmetry_transform_filament1(geom, args.sym_distance)` as `T_fi`
- Else if `--optimize-filaments`: run the existing scan/Brent optimisation, use `T_rot` as `T_fi`
- Else: skip this filament

After the loop: compose `chain_extra_rot` into `chain_transforms` (same logic as today). Set `filament_rotations = None` if nothing was added.

This means:
- `--optimize-filaments` alone → both filaments optimised (same as today)
- `--enforce-symmetry` alone → filament 1 symmetrised, filament 0 untouched
- Both → filament 0 optimised, filament 1 symmetrised

---

**Verification**
1. `python script.py --save-candidates camkii_candidates_v4` — geometry printed, no crash
2. `python script.py --assignment camkii_assignment.txt --enforce-symmetry` — T_sym applied, PDB written
3. `python script.py --assignment camkii_assignment.txt --optimize-filaments --enforce-symmetry` — fil0 optimised, fil1 symmetrised
4. VMD check: filament axes parallel, 240 Å apart