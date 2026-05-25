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

### 3. Animation

Add `--animate N` to either mode to also write an N-frame multi-model PDB:

```bash
python script.py --assignment camkii_assignment.txt --animate 20
```

This produces `camkii_on_actin_movie.pdb` (≈10 MB per frame, so ~200 MB at
N=20). Open it as a trajectory:

| Viewer | Open as |
| ------ | ------- |
| PyMOL | `load camkii_on_actin_movie.pdb, mov, multiplex=0` |
| VMD | `mol new camkii_on_actin_movie.pdb` (auto-detected) |
| ChimeraX | `open camkii_on_actin_movie.pdb` (auto-detected) |

Hub and actins stay fixed across frames; only the 12 kinase domains and
their linkers move.

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
5. **Move kinases + morph linkers.** For each assigned source chain,
   `compute_transformation()` puts the source kinase onto the candidate,
   `transform()` applies that to the kinase residues, and
   `morph_segment(method='sclerp')` ScLERP-interpolates the linker between
   the unchanged hub end (`identity`) and the moved kinase end (`T_kin`).
6. **Write** the final PDB (and the movie if `--animate` was given).

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
