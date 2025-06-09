import os
import subprocess
import tempfile
import logging
from typing import Union

logger = logging.getLogger(__name__)

def count_atoms_with_vmd(pdb_paths: list[str], selections: list[str]) -> dict[tuple[str, str], int]:
    """
    Count atoms for multiple PDB files and multiple selections in a single VMD call.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    selections : list of str
        List of VMD atom-selection strings.

    Returns
    -------
    dict
        Mapping from (pdb_path, selection) to atom count.

    Raises
    ------
    RuntimeError
        If VMD fails or output cannot be parsed.
    """
    logger.info(f"Calling VMD to count atoms in {pdb_paths} with selections {selections}")
    logger.debug(f"Preparing Tcl script for VMD selections.")

    # Build Tcl script to load each PDB and count atoms for each selection
    tcl_lines = []
    for pdb in pdb_paths:
        tcl_lines.append(f"mol new \"{os.path.abspath(pdb)}\"")
        for sel in selections:
            tcl_lines.append(f"set sel [atomselect top \"{sel}\"]")
            tcl_lines.append(f"puts \"COUNT {os.path.basename(pdb)} | {sel} | [$sel num]\"")
            tcl_lines.append("$sel delete")
        tcl_lines.append("mol delete top")
    tcl_lines.append("exit")
    tcl = "\n".join(tcl_lines)

    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as script:
        script.write(tcl)
        script_path = script.name

    logger.debug(f"Temporary Tcl script written to: {script_path}")
    logger.debug(f"Tcl script contents:\n{tcl}")

    cmd = [
        "vmd",
        "-dispdev", "text",
        "-e", script_path
    ]
    logger.info(f"Calling VMD with command: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    logger.debug(f"VMD return code: {proc.returncode}")
    logger.debug(f"VMD stdout:\n{proc.stdout}")
    logger.debug(f"VMD stderr:\n{proc.stderr}")

    if proc.returncode != 0:
        logger.error(f"VMD failed (rc={proc.returncode}):\n{proc.stderr.strip()}")
        raise RuntimeError(f"VMD failed (rc={proc.returncode}):\n{proc.stderr.strip()}")

    # Parse output lines like: COUNT filename | selection | N
    result = {}
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("COUNT "):
            try:
                _, rest = s.split("COUNT ", 1)
                pdbfile, sel, n = [x.strip() for x in rest.split("|")]
                result[(pdbfile, sel)] = int(n)
                logger.info(f"Found atom count: {pdbfile} | {sel} = {n}")
            except Exception as e:
                logger.warning(f"Failed to parse line: {s} ({e})")
    if not result:
        logger.error("Could not parse VMD output for atom counts.")
        raise RuntimeError(
            "Could not parse VMD output for atom counts.\n"
            f"Full stdout:\n{proc.stdout}\n"
            f"Full stderr:\n{proc.stderr}"
        )
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    n = count_atoms_with_vmd(["molscene/data/1r70.pdb", "molscene/data/1zbl.cif"], ["protein","nucleic","water"])
    print(f"Number of atoms: {n}")