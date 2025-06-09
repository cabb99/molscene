import os
import subprocess
import tempfile
import logging
from typing import Union

# Set up logging with multiple levels
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_atoms_with_vmd(pdb_path: str, selection: str) -> int:
    """
    Count atoms in `pdb_path` matching the VMD selection string `selection`
    by running VMD in text mode.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    selection : str
        A VMD atom‐selection string, e.g. "resid 1" or "protein".

    Returns
    -------
    int
        Number of atoms matching the selection.

    Raises
    ------
    RuntimeError
        If VMD fails or no numeric output can be parsed.
    """
    logger.info(f"Counting atoms in '{pdb_path}' with selection '{selection}'")
    logger.debug(f"Preparing Tcl script for VMD selection.")

    # Create a tiny Tcl script on disk
    tcl = f"""
set sel [atomselect top \"{selection}\"]
puts [$sel num]
exit
"""
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as script:
        script.write(tcl)
        script_path = script.name

    logger.debug(f"Temporary Tcl script written to: {script_path}")
    logger.debug(f"Tcl script contents:\n{tcl}")

    # Build the VMD command: text mode, execute our script, pass PDB as $argv0
    abs_pdb = os.path.abspath(pdb_path)
    cmd = [
        "vmd",
        "-dispdev", "text",
        "-e", script_path,
        abs_pdb
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

    # Scan output for the first line that's just digits
    for line in proc.stdout.splitlines():
        s = line.strip()
        logger.debug(f"Scanning VMD output line: '{s}'")
        if s.isdigit():
            logger.info(f"Found atom count in VMD output: {s}")
            return int(s)

    logger.error("Could not parse VMD output for atom count.")
    raise RuntimeError(
        "Could not parse VMD output for atom count.\n"
        f"Full stdout:\n{proc.stdout}\n"
        f"Full stderr:\n{proc.stderr}"
    )


if __name__ == "__main__":
    n = count_atoms_with_vmd("molscene/data/1r70.pdb", "protein")
    print(f"Number of atoms: {n}")