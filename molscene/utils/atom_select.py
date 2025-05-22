import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from lark import Lark, Transformer, v_args

# ————————————————————————————————————————————————————————————————————
# 1) Lark grammar capturing the ENTIRE VMD selection language
# ————————————————————————————————————————————————————————————————————
with open("molscene/utils/selection_syntax.lark", "r") as f:
    Lark(f.read(), parser='lalr', debug=True)

_protein_residues = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                     'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                     'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                     'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                     'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

_DNA_residues = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}

_RNA_residues = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}

# ————————————————————————————————————————————————————————————————————
# 2) Transformer: AST → Boolean masks over a Scene
# ————————————————————————————————————————————————————————————————————
@v_args(inline=True)
class VMDTransformer(Transformer):
    def __init__(self, scene, macros: dict, variables: dict):
        self.scene    = scene
        self.df       = scene    # alias
        self.macros   = macros   # @macro → selstr
        self.vars     = variables  # $var → array or scalar
        # pre‐compute these for speed:
        self._coords  = scene.get_coordinates().to_numpy()
        self._tree    = cKDTree(self._coords)
        self.prot_res = set(_protein_residues.keys())
        self.dna_res  = set(_DNA_residues.keys())
        self.rna_res  = set(_RNA_residues.keys())
        self.backbone = {"N","CA","C","O"}  # classic VMD backbone
        super().__init__()

    # — atomic‐flag keywords
    def flag_sel(self, name):
        key = str(name)
        df = self.df
        key_l = key.lower()

        if key_l == "protein":
            return df["resName"].isin(self.prot_res)
        if key_l == "nucleic":
            return df["resName"].isin(self.dna_res|self.rna_res)
        if key_l == "backbone":
            return df["name"].isin(self.backbone)
        if key_l == "water":
            return df["resName"].isin({"HOH","WAT"})
        if key_l == "hetero":
            ok = set(df["resName"].unique())
            allowed = self.prot_res | self.dna_res | self.rna_res | {"HOH","WAT"}
            return ~df["resName"].isin(allowed)
        if key_l == "resid":
            # resid == resSeq
            return df["resSeq"]
        if key_l == "index":
            return df.index
        # numeric atom fields or string fields—just defer to pandas:
        # The parser will end up calling DataFrame.eval on them.
        return key

    # — single‐value or  “name CA PHE …”
    def range_sel(self, key, *vals):
        k = str(key)
        # flatten vals, ranges:
        tokens = []
        for v in vals:
            if isinstance(v, tuple):
                lo, hi = v
                tokens.append( (float(lo), float(hi)) )
            else:
                tokens.append( v )
        # numeric vs string: inspect dtype
        col = self.df[k]
        if pd.api.types.is_numeric_dtype(col):
            mask = pd.Series(False, index=col.index)
            for t in tokens:
                if isinstance(t, tuple):
                    lo, hi = t
                    mask |= (col>=lo)&(col<=hi)
                else:
                    mask |= col==float(t)
            return mask
        else:
            # all tokens that are not ranges:
            vals = [str(t) for t in tokens if not isinstance(t, tuple)]
            return col.isin(vals)

    def single_value(self, tok):
        s = tok.value
        if re.match(r"^'|^\"", s):
            return s.strip("'\"")
        return s

    def range_value(self, lo, _, hi):
        return (lo, hi)

    # — regex in double quotes
    def regex_sel(self, key, regex):
        pat = regex.value.strip('"')
        return self.df[key].str.match(pat, na=False)

    # — numeric/string comparisons
    def comparison(self, left, op, right):
        # if left or right is a bare column name (i.e. str), we defer to DataFrame.eval
        l = left if not isinstance(left, str) else f"`{left}`"
        r = right if not isinstance(right, str) else f"`{right}`"
        expr = f"{l} {op} {r}"
        return self.df.eval(expr)

    def cmp_op(self, tok):
        return tok.value

    # — functions
    def func_call(self, name, arg):
        name = str(name)
        f = {
            "abs": np.abs,
            "sqrt": np.sqrt,
            "sq":  lambda x: x**2,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin":np.arcsin,
            "acos":np.arccos,
            "atan":np.arctan,
            "floor":np.floor,
            "ceil": np.ceil,
            "sinh":np.sinh,
            "cosh":np.cosh,
            "tanh":np.tanh
        }[name]
        return f(arg)

    # — within / exwithin
    def within_sel(self, dist, _, of_sel):
        d = float(dist)
        idx2 = np.where(of_sel)[0]
        ds, _ = self._tree.query(self._coords, k= len(idx2), n_jobs=-1)
        # but we only need the min distance to ANY of idx2:
        # simpler to rebuild a small tree:
        t2 = cKDTree(self._coords[idx2])
        minD,_ = t2.query(self._coords, n_jobs=-1)
        return pd.Series(minD <= d, index=self.df.index)

    def exwithin_sel(self, dist, _, of_sel):
        return ~self.within_sel(dist, _, of_sel)

    # — same <keyword> as <sel>
    def same_sel(self, _, key, __, sel):
        col = self.df[str(key)]
        vals = set( col[sel] )
        return col.isin(vals)

    # — sequence "PATTERN"
    def sequence_sel(self, _, regex):
        pat = regex.value.strip('"')
        out = pd.Series(False, index=self.df.index)
        for chain, grp in self.df.groupby("chainID"):
            # build one‐letter seq
            seq = []
            residues = []
            for resSeq, sub in grp.groupby("resSeq"):
                aa = sub["resName"].iat[0]
                seq.append( _protein_residues.get(aa, 
                          _DNA_residues.get(aa,
                          _RNA_residues.get(aa, "X"))) )
                residues.append(resSeq)
            S = "".join(seq)
            for m in re.finditer(pat, S):
                # select all atoms in those matching residues
                for resSeq in residues[m.start():m.end()]:
                    out[ (self.df["chainID"]==chain)&(self.df["resSeq"]==resSeq) ] = True
        return out

    # — macros & user vars
    def macro_sel(self, name):
        selstr = self.macros[str(name)]
        return Scene(self.scene).select_vmd(selstr, **self.vars).index

    def var_sel(self, name):
        return self.vars[str(name)]

    # — combining boolean ops
    def and_(self, a, b): return a & b
    def or_(self, a, b):  return a | b
    def not_(self, a):     return ~a

    # leaves numbers as floats or ints
    def SIGNED_NUMBER(self, t):
        return float(t)

    def CNAME(self, t):
        return str(t)

# ————————————————————————————————————————————————————————————————————
# 3) The public API: Scene.select_vmd
# ————————————————————————————————————————————————————————————————————
class VMDSelector:
    with open("molscene/utils/selection_syntax.lark", "r") as f:
        parser = Lark(f.read(), parser='lalr', debug=True)

    def __init__(self, scene, macros=None, **variables):
        self.scene  = scene
        self.macros = macros or {}
        self.vars    = variables

    def __call__(self, selstr: str) -> pd.Index:
        tree = self._parser.parse(selstr)
        mask = VMDTransformer(self.scene, self.macros, self.vars).transform(tree)
        # mask is a boolean Series
        return mask

if __name__ == "__main__":
    from lark import exceptions
    
    # Examples classified by feature/difficulty
    EXAMPLES = [
        # --- Simple flags and terms ---
        ("Simple flag: protein", "protein"),
        ("Simple flag: water", "water"),
        ("Simple field: name CA", "name CA"),
        ("Multiple names", "name CA CB"),
        ("Multiple resnames", "resname ALA GLY"),
        ("Residue id", "resid 4"),
        ("Index", "index 5"),
        ("Backbone", "backbone"),
        ("Hetero", "hetero"),
        ("All atoms", "all"),
        ("None atoms", "none"),
        ("Waters alias", "waters"),
        ("Is_protein alias", "is_protein"),
        ("Is_water alias", "is_water"),
        ("Everything alias", "everything"),
        ("Nothing alias", "nothing"),
        ("Name with quotes", "name 'CA'"),
        ("Name with quotes 2", 'name "CA"'),
        ("Name with quotes 3", 'name "CA" "CB" "CA CB"'),

        # --- Logic and default AND ---
        ("AND logic", "protein and water"),
        ("OR logic", "protein or water"),
        ("NOT logic", "not water"),
        ("Default AND", "not water hetero"),
        ("Default AND with fields", "resname ALA PHE name CA CB"),
        ("Default AND with flags", "acidic calpha"),
        ("Parentheses", "(protein or water) and not hetero"),
        ("Nested NOT", "not not (protein and water)"),
        ("NOT with !", "!protein"),

        # --- Numeric comparisons and ranges ---
        ("Numeric comparison", "mass>12"),
        ("Numeric comparison with AND", "mass>12 and mass<17"),
        ("Range selection", "resid 10 to 20"),
        ("Range selection with AND", "resid 10 to 20 and backbone"),
        ("Python-style range", "resid 1:4"),
        ("Python-style range with step", "resid 1:4:2"),
        ("Negative number range", "x -25 to 25"),
        ("Negative number single", "x -22.542"),
        ("Chained comparison", "-10 <= x < 0"),
        ("Comparison with math", "x ** 2 < 10"),
        ("Comparison with function", "sqrt(sq(x) + sq(y) + sq(z)) < 100"),
        ("Element selection", "element O"),
        ("Mass range float", "mass 5.5 to 12.3"),
        ("Binary selection operator", "name < 1"),
        ("Binary selection operator (ne)", "name ne O"),
        ("Binary selection operator (eq)", "name eq 1"),
        ("Binary selection operator (ge)", "name ge 1"),
        ("Binary selection operator (le)", "name le 1"),
        ("Binary selection operator (gt)", "name gt 1"),
        ("Binary selection operator (lt)", "name lt 1"),
        ("Reverse binary selection", "1 == name"),
        ("Reverse binary selection (eq)", "1 eq name"),

        # --- Regular expressions ---
        ("Regex on resname", 'resname "A.*"'),
        ("Regex on sequence", 'sequence "MI.*DKQ"'),
        ("Regex on name", 'name =~ "C.*"'),
        ("Regex on name with AND", '(name =~ "C.*") and all'),

        # --- Distance-based selections ---
        ("Within distance", "within 5 of water"),
        ("Exwithin distance", "exwithin 3 of water"),
        ("Within distance of field", "within 5 of name FE"),
        ("Within with parentheses", "within 5 of (backbone or sidechain)"),

        # --- Sequence and macros ---
        ("Sequence pattern", 'sequence "C..C"'),
        ("Macro usage", "@alanine"),
        ("Macro with AND", "protein and @alanine"),
        ("User variable", "within 8 of $center and @alanine"),

        # --- Parentheses and complex logic ---
        ("Parentheses", "(protein or water) and not hetero"),
        ("Same residue as", "same residue as exwithin 4 of water"),
        ("Same resname as", "same resname as (protein within 6 of nucleic)"),
        ("Complex logic", "protein or water or all"),
        ("Nested bool", "nothing and water or all"),
        ("Nested bool with parens", "nothing and (water or all)"),
        ("Quotes test 1", "name CA and resname ALA"),
        ("Quotes test 2", 'name "CA" and resname ALA'),
        ("Quotes test 3", "name 'CA' and resname ALA"),
        ("In operator with resname", "resname ALA ASP GLU"),
        ("In operator with resid", "resid 100 101 102"),

        # ---Complex selections ---
        ("Complex selection", "protein and (resname ALA or resname GLY) and not water"),
        ("Complex selection with parentheses", "(protein or water) and not hetero"),
        ("Complex selection with regex", 'resname "A.*" and name =~ "C.*"'),
        ("Complex selection with distance", "protein within 5 of (resname ALA or resname GLY)"),
        ("Complex selection with sequence", 'sequence "C..C" and resname ALA'),
        ("Complex selection with macro", "@alanine and within 5 of water"),
        ("Complex selection with user variable", "$var1 and $var2"),
        ("Complex selection with function", "sqrt(x**2 + y**2 + z**2) < 10"),

        # --- Math ---
        ("Math with functions", "sqrt(x**2 + y**2 + z**2) < 10"),
        ("Math with abs", "abs(x) < 5"),
        ("Math with log", "log(x) > 0"),
        ("Math with exp", "exp(x) < 100"),
        ("Math with sin", "sin(x) > 0.5"),
        ("Math with cos", "cos(x) < 0.5"),
        ("Math with tan", "tan(x) > 1"),
        ("Complex math with functions", "sqrt(z^3-sin(x*y)^2) < 10"),




    ]

    with open("molscene/utils/selection_syntax.lark", "r") as f:
        parser = Lark(f.read(), parser='lalr', debug=True)
    for i, (desc, sel) in enumerate(EXAMPLES):
        print(f"\nExample {i+1} [{desc}]: {sel}")
        try:
            tree = parser.parse(sel)
            print("Parse tree:")
            print(tree.pretty())
        except exceptions.LarkError as e:
            print(f"Parse failed: {e.__class__.__name__}: {e}")
