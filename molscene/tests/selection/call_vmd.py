import os
import subprocess
import tempfile
import logging
from typing import Union
import numpy as np
import json
import concurrent.futures

logger = logging.getLogger(__name__)

def count_atoms_with_vmd(pdb_paths: list[str], selections: list[str], tcl_script_path: str | None = None) -> dict[tuple[str, str], int]:
    """
    Count atoms for multiple PDB files and multiple selections in a single VMD call.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    selections : list of str
        List of VMD atom-selection strings.
    tcl_script_path : str or None, optional
        If provided, write the generated Tcl script to this file instead of a temporary file.

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

    delimiter = '|----|' # Use a unique delimiter unlikely to appear in filenames or selections
    # Build Tcl script to load each PDB and count atoms for each selection
    tcl_lines = []
    for pdb in pdb_paths:
        tcl_lines.append(f"mol new \"{os.path.abspath(pdb)}\"")
        for sel in selections:
            escaped_sel = (
                sel
                .replace('\\', '\\\\')   # escape backslashes first
                .replace('"', '\\"')     # escape double-quotes
                .replace('$', '\\$')     # escape dollar signs (no $VAR substitution)
                .replace('[', '\\[')     # escape open brackets (no [cmd] substitution)
                .replace(']', '\\]')     # (optional) escape closing brackets for symmetry
            )
            # Use try to catch errors in atomselect and puts
            tcl_lines.append(f"try {{")
            tcl_lines.append(f'    set sel [atomselect top "{escaped_sel}"]')
            tcl_lines.append(f'    puts "COUNT {os.path.basename(pdb)}{delimiter}{escaped_sel}{delimiter}[$sel num]"')
            tcl_lines.append(f"    $sel delete")
            tcl_lines.append(f"}} on error {{err opts}} {{")
            tcl_lines.append(f'    puts "COUNT {os.path.basename(pdb)}{delimiter}{escaped_sel}{delimiter}nan"')
            tcl_lines.append(f"}}")
        tcl_lines.append("mol delete top")
    tcl_lines.append("exit")
    tcl = "\n".join(tcl_lines)

    if tcl_script_path is not None:
        with open(tcl_script_path, "w") as script:
            script.write(tcl)
        script_path = tcl_script_path
        logger.debug(f"Tcl script written to user-specified file: {script_path}")
    else:
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

    # Parse output lines like: COUNT filename<<<DELIM>>>selection<<<DELIM>>>N
    result = {}
    found_keys = set()
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("COUNT "):
            try:
                _, rest = s.split("COUNT ", 1)
                parts = rest.split(delimiter)
                if len(parts) != 3:
                    logger.warning(f"Unexpected output format: {s}")
                    continue
                pdbfile, sel, n = [x.strip() for x in parts]
                found_keys.add((pdbfile, sel))
                if n.isdigit():
                    result[(pdbfile, sel)] = int(n)
                    logger.info(f"Found atom count: {pdbfile} | {sel} = {n}")
                else:
                    result[(pdbfile, sel)] = np.nan
                    logger.info(f"Invalid selection or count for: {pdbfile} | {sel}, returning np.nan")
            except Exception as e:
                logger.warning(f"Failed to parse line: {s} ({e})")
    # Fill missing (pdb, sel) pairs with np.nan
    for pdb in pdb_paths:
        for sel in selections:
            key = (os.path.basename(pdb), sel)
            if key not in result:
                logger.warning(f"Missing count for {key}, setting to np.nan")
                result[key] = np.nan
    if not result:
        logger.error("Could not parse VMD output for atom counts.")
        raise RuntimeError(
            "Could not parse VMD output for atom counts.\n"
            f"Full stdout:\n{proc.stdout}\n"
            f"Full stderr:\n{proc.stderr}"
        )
    return result


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def _prody_select_worker(args):
    pdb_basename, sel, pdb_path = args
    import prody
    import numpy as np
    try:
        structure = prody.parsePDB(pdb_path)
        atoms = structure.select(sel)
        count = len(atoms) if atoms is not None else 0
        return (pdb_basename, sel, count)
    except Exception:
        return (pdb_basename, sel, np.nan)

def count_atoms_with_prody(pdb_paths: list[str], selections: list[str], max_workers: int = 4) -> dict[tuple[str, str], int]:
    """
    Count atoms for multiple PDB files and multiple selections using ProDy, with multiprocessing.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    selections : list of str
        List of ProDy atom-selection strings.
    max_workers : int
        Number of worker processes to use (default: 4).

    Returns
    -------
    dict
        Mapping from (pdb_path, selection) to atom count (np.nan if selection fails).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(pdb_paths)} PDBs x {len(selections)} selections with up to {max_workers} workers.")
    tasks = []
    for pdb in pdb_paths:
        pdb_basename = os.path.basename(pdb)
        for sel in selections:
            tasks.append((pdb_basename, sel, pdb))
    result = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, (pdb_basename, sel, count) in enumerate(executor.map(_prody_select_worker, tasks)):
            result[(pdb_basename, sel)] = count
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i} selections...")
    return result


if __name__ == "__main__":
    import glob
    import pandas as pd

    logging.basicConfig(level=logging.INFO)  # or logging.DEBUG for more verbosity
    
    # Read and clean the JSONC file (remove C++ style comments)
    jsonc_path = os.path.join(os.path.dirname(__file__), "selection_tests.jsonc")
    with open(jsonc_path, "r") as f:
        lines = f.readlines()
    # Only skip lines that start with // (do not strip inline comments)
    clean_lines = []
    for line in lines:
        if line.lstrip().startswith("//"):  # skip full-line comments
            continue
        clean_lines.append(line)
    clean_json = "".join(clean_lines)
    selection_tests = json.loads(clean_json)
    selections = [test["query"] for test in selection_tests]
    print(f"Loaded {len(selections)} selection queries from {jsonc_path}")
    pdb_files = glob.glob('molscene/data/*.pdb') + glob.glob('molscene/data/*.cif')
    
    n_vmd = count_atoms_with_vmd(pdb_files, selections, tcl_script_path='temp_vmd_script.tcl')
    n_prody = count_atoms_with_prody(pdb_files, selections)

    # Merge results into a pandas DataFrame
    df_vmd = pd.DataFrame([
        {"pdb": pdb, "selection": sel, "count_vmd": count}
        for (pdb, sel), count in n_vmd.items()
    ])
    df_prody = pd.DataFrame([
        {"pdb": pdb, "selection": sel, "count_prody": count}
        for (pdb, sel), count in n_prody.items()
    ])
    df = pd.merge(df_vmd, df_prody, on=["pdb", "selection"], how="outer")

    print(f"Number of atoms (per selection):")
    print(df.to_string(index=False, max_colwidth=80))