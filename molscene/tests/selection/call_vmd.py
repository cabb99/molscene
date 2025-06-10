import os
import subprocess
import tempfile
import logging
from typing import Union
import numpy as np
import json

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
    from prody import parsePDB
    import numpy as np
    try:
        structure = parsePDB(pdb_path)
        atoms = structure.select(sel)
        count = len(atoms) if atoms is not None else 0
        return (pdb_basename, sel, count)
    except Exception:
        return (pdb_basename, sel, np.nan)

def count_atoms_with_prody(pdb_paths: list[str], selections: list[str]) -> dict[tuple[str, str], int]:
    """
    Count atoms for multiple PDB files and multiple selections using ProDy.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    selections : list of str
        List of ProDy atom-selection strings.

    Returns
    -------
    dict
        Mapping from (pdb_path, selection) to atom count (np.nan if selection fails).
    """
    from prody import parsePDB
    logger = logging.getLogger(__name__)
    result = {}
    for pdb in pdb_paths:
        pdb_basename = os.path.basename(pdb)
        logger.info(f"Processing PDB: {pdb_basename} with {len(selections)} selections")
        try:
            structure = parsePDB(pdb)
        except Exception as e:
            logger.warning(f"Failed to parse PDB: {pdb}: {e}")
            for sel in selections:
                result[(pdb_basename, sel)] = np.nan
            continue
        for i, sel in enumerate(selections):
            try:
                atoms = structure.select(sel)
                count = len(atoms) if atoms is not None else 0
                result[(pdb_basename, sel)] = count
            except Exception as e:
                if i < 5:
                    logger.info(f"[ProDy] Selection failed for '{sel}': {e}")
                result[(pdb_basename, sel)] = np.nan
            if i % 100 == 0 and i > 0:
                logger.info(f"[ProDy] Processed {i} selections for {pdb_basename}")
    return result

def count_atoms_with_molscene(pdb_paths: list[str], selections: list[str]) -> dict[tuple[str, str], int]:
    """
    Count atoms for multiple PDB files and multiple selections using the molscene transformer parser.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    selections : list of str
        List of selection queries.

    Returns
    -------
    dict
        Mapping from (pdb_path, selection) to atom count (np.nan if selection fails).
    """
    import pandas as pd
    import numpy as np
    import os
    import logging
    from molscene.selection import transformer as molscene_transformer

    logger = logging.getLogger(__name__)
    result = {}
    for pdb in pdb_paths:
        pdb_basename = os.path.basename(pdb)
        logger.info(f"Processing PDB with molscene: {pdb_basename} with {len(selections)} selections")
        try:
            # Use Scene.from_pdb or Scene.from_cif to load structure as DataFrame
            if pdb.endswith('.pdb'):
                from molscene.Scene import Scene
                df = Scene.from_pdb(pdb)
            elif pdb.endswith('.cif'):
                from molscene.Scene import Scene
                df = Scene.from_cif(pdb)
            else:
                logger.warning(f"Unsupported file format for {pdb}")
                for sel in selections:
                    result[(pdb_basename, sel)] = np.nan
                continue
            # Build parser and transformer
            grammar_path = os.path.join(os.path.dirname(molscene_transformer.__file__), "selection_syntax.lark")
            with open(grammar_path) as f:
                grammar_text = f.read()
            from lark import Lark
            parser = Lark(grammar_text, parser='lalr', propagate_positions=True, start=['start', 'expr'])
            transformer = molscene_transformer.PandasTransformer(df, parser=parser)
            for sel in selections:
                try:
                    tree = parser.parse(sel, start='start')
                    selected = transformer.transform(tree)
                    count = len(selected)
                    result[(pdb_basename, sel)] = count
                except Exception as e:
                    logger.info(f"[molscene] Selection failed for '{sel}': {e}")
                    result[(pdb_basename, sel)] = np.nan
        except Exception as e:
            logger.warning(f"molscene transformer failed for {pdb}: {e}")
            for sel in selections:
                result[(pdb_basename, sel)] = np.nan
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
    
    n_molscene = count_atoms_with_molscene(pdb_files, selections)
    n_vmd = count_atoms_with_vmd(pdb_files, selections, tcl_script_path='temp_vmd_script.tcl')
    n_prody = count_atoms_with_prody(pdb_files, selections)

    # Merge results into a pandas DataFrame
    df_molscene = pd.DataFrame([
        {"pdb": pdb, "selection": sel, "count_molscene": count}
        for (pdb, sel), count in n_molscene.items()
    ])
    df_vmd = pd.DataFrame([
        {"pdb": pdb, "selection": sel, "count_vmd": count}
        for (pdb, sel), count in n_vmd.items()
    ])
    df_prody = pd.DataFrame([
        {"pdb": pdb, "selection": sel, "count_prody": count}
        for (pdb, sel), count in n_prody.items()
    ])

    df = df_vmd
    df = pd.merge(df_vmd, df_prody, on=["pdb", "selection"], how="outer")
    df = pd.merge(df, df_molscene, on=["pdb", "selection"], how="outer")

    print(f"Number of atoms (per selection):")
    print(df.to_string(index=False, max_colwidth=80))

    df.to_csv("atom_countsv2.csv", index=False)