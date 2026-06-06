"""``awsem_gro`` format — the AWSEM fragment-memory structure file.

This is the bespoke "Structure-Based gro file" that OpenAWSEM's fragment-memory
term consumes.  It is *not* a real GROMACS gro (no box line, custom 2-line header,
coordinates in nm) — molscene calls the format ``awsem_gro``.  This module holds
the reader, the writer (byte-compatible with OpenAWSEM's ``Pdb2Gro``) and the
``.mem`` single-memory index helper, all Scene-agnostic.
"""

import os
from pathlib import Path

import pandas

from .registry import FormatRegistry

# Atoms required for a residue to count as a "regular" protein residue.
_AWSEM_BACKBONE = {"N", "CA", "C"}


def _residue_groups(frame):
    """Yield ``(chain, resid, group)`` for regular protein residues, in order.

    A residue qualifies when it carries the full N/CA/C backbone (this excludes
    water, ions and other heterogens).  ``frame`` is a plain DataFrame in atom
    order; grouping is stable so chains/residues keep their order.
    """
    for (chain, resid, _icode), group in frame.groupby(
            ["chain", "resid", "icode"], sort=False):
        if _AWSEM_BACKBONE <= set(group["name"]):
            yield chain, resid, group


def _gro_line(res_no, res_name, atom_name, atom_no, x, y, z):
    """Format one atom line, byte-compatible with OpenAWSEM ``Pdb2Gro.Atom.write_``
    (fixed-width fields; coordinates Angstrom -> nm)."""
    def w8(v):
        return ("        " + str(round(v / 10, 3)))[-8:]
    return (("     " + str(res_no))[-5:]
            + ("     " + str(res_name))[-5:]
            + " " + (str(atom_name) + "    ")[:4]
            + ("     " + str(atom_no))[-5:]
            + w8(x) + w8(y) + w8(z) + "\n")


def chain_lengths(scene, chain=None):
    """Ordered ``[(chain, n_residues), ...]`` of regular protein residues."""
    frame = pandas.DataFrame(scene)
    if chain is not None:
        chains = [chain] if isinstance(chain, str) else list(chain)
        frame = frame[frame["chain"].isin(chains)]
    counts = {}
    for ch, _resid, _group in _residue_groups(frame):
        counts[ch] = counts.get(ch, 0) + 1
    return list(counts.items())


@FormatRegistry.register_writer(".gro", name="awsem_gro")
def write_awsem_gro(scene, file_name=None, chain=None):
    """Write an AWSEM fragment-memory ``.gro``.

    Only regular protein residues (those with the N/CA/C backbone) are written;
    alternate locations are collapsed by occupancy and atom numbering restarts per
    chain, as in OpenAWSEM's ``Pdb2Gro``.  ``file_name=None`` returns the text.
    """
    frame = pandas.DataFrame(scene).reset_index(drop=True)
    if chain is not None:
        chains = [chain] if isinstance(chain, str) else list(chain)
        frame = frame[frame["chain"].isin(chains)]

    # Collapse alternate locations: keep one atom per (chain, resid, icode, name),
    # preferring the highest occupancy, without disturbing atom order.
    keys = ["chain", "resid", "icode", "name"]
    if "occupancy" in frame.columns:
        frame = frame.sort_values("occupancy", ascending=False, kind="stable")
    frame = frame.drop_duplicates(subset=keys, keep="first").sort_index(kind="stable")

    lines = []
    current_chain = None
    iatom = 0
    for ch, resid, group in _residue_groups(frame):
        if ch != current_chain:
            current_chain = ch
            iatom = 0
        res_name = group["resname"].iloc[0]
        for _, atom in group.iterrows():
            iatom += 1
            lines.append(_gro_line(resid, res_name, atom["name"], iatom,
                                   atom["x"], atom["y"], atom["z"]))

    text = " Structure-Based gro file\n" + f"{len(lines):12}\n" + "".join(lines)
    if file_name is None:
        return text
    with open(file_name, "w") as fh:
        fh.write(text)


@FormatRegistry.register_reader(".gro", name="awsem_gro")
def read_awsem_gro(path, **kw):
    """Parse an ``awsem_gro`` file into ``(atoms_DataFrame, meta)``.

    Inverse of :func:`write_awsem_gro`: skip the 2-line header, read the 7
    whitespace columns ``res_no res_name atom_name atom_no x y z`` and convert
    coordinates nm -> Angstrom.  The format carries no chain, so ``chain='A'``.
    """
    rows = []
    with open(path) as fh:
        body = fh.read().splitlines()[2:]   # skip the 2-line header
    for line in body:
        parts = line.split()
        if len(parts) < 7:
            continue
        res_no, res_name, atom_name, atom_no, x, y, z = parts[:7]
        rows.append(dict(
            recname="ATOM", serial=int(atom_no), name=atom_name, resname=res_name,
            chain="A", resid=int(res_no), icode="",
            x=float(x) * 10, y=float(y) * 10, z=float(z) * 10))
    atoms = pandas.DataFrame(rows)
    meta = {"source_file": str(Path(path).resolve()), "source_format": "awsem_gro"}
    return atoms, meta


# --- fragment-memory ``.mem`` index ---------------------------------------------

def memory_line(gro, target_start, length, weight=20, gro_start=1):
    """One ``[Memories]`` record: ``<gro> <target_start> <gro_start> <length> <weight>``."""
    return f"{gro} {target_start} {gro_start} {length} {weight}"


def write_single_memory(scene, name="query", directory=".", weight=20,
                        mem_filename="single_frags.mem"):
    """Write a single-memory set for ``scene`` (the structure *is* the target).

    For every chain with regular protein residues this writes ``<name>_<chain>.gro``
    and adds a ``[Memories]`` line; the gro residue index always starts at 1 and the
    target start is the cumulative residue offset across chains.  Returns the ``.mem``
    path.
    """
    os.makedirs(directory, exist_ok=True)
    body = []
    target_start = 1
    for chain, length in chain_lengths(scene):
        if length == 0:
            continue
        gro_name = f"{name}_{chain}.gro"
        write_awsem_gro(scene, os.path.join(directory, gro_name), chain=chain)
        body.append(memory_line(gro_name, target_start, length, weight=weight))
        target_start += length

    mem_path = os.path.join(directory, mem_filename)
    with open(mem_path, "w") as fh:
        fh.write("[Target]\nquery\n\n[Memories]\n")
        for line in body:
            fh.write(line + "\n")
    return mem_path
