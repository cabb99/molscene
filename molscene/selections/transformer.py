import numpy as np
import math
import pandas as pd
from lark import Lark, Transformer, v_args
import logging
import json
import os

@v_args(inline=True)
class PandasTransformer(Transformer):
    from operator import add, sub, mul, truediv, floordiv, mod, pow, neg, and_, or_, xor, lt, gt, le, ge, eq, ne
    
    def _load_and_flatten_macros(self, macros_path):
        """Load macros from macros.json and flatten them into a {name: definition} dict, including synonyms."""
        macros_flat = {}
        if os.path.exists(macros_path):
            with open(macros_path, 'r') as f:
                try:
                    macros_json = json.load(f)
                    for category in macros_json.get('macros', {}):
                        for macro_name, macro_obj in macros_json['macros'][category].items():
                            definition = macro_obj.get('definition', '')
                            macros_flat[macro_name] = definition
                            # Add synonyms
                            for syn in macro_obj.get('synonyms', []):
                                macros_flat[syn] = definition
                except Exception as e:
                    logging.warning(f"Failed to load macros from {macros_path}: {e}")
        return macros_flat

    def __init__(self, df, macros=None, parser=None):
        super().__init__()
        self.df = df
        self.macros = {}
        if macros is None:
            macros = {}
        self.parser = parser
        default_macros_file = os.path.join(os.path.dirname(__file__), 'macros.json')
        default_macros = self._load_and_flatten_macros(default_macros_file)
        self.macros = {**default_macros, **macros}
        logging.debug(f"PandasTransformer initialized with DataFrame of shape {df.shape} and macros: {list(self.macros.keys())}")

    def add_macro(self, name, expr):
        """Add or update a macro."""
        self.macros[name] = expr
        logging.info(f"Macro '{name}' added/updated.")

    def remove_macro(self, name):
        """Remove a macro if it exists."""
        if name in self.macros:
            del self.macros[name]
            logging.info(f"Macro '{name}' removed.")
        else:
            logging.warning(f"Macro '{name}' not found for removal.")

    def start(self, mask):
        logging.debug(f"start() called with mask of type {type(mask).__name__}: {repr(mask)}")
        return self.df[mask]
    
    def select_all(self):
        # a Series of True, one entry per row
        return pd.Series(True, index=self.df.index)

    def select_none(self):
        # a Series of False, one entry per row
        return pd.Series(False, index=self.df.index)

    def not_(self, mask):
        logging.debug(f"not_() called with mask: {repr(mask)}")
        return ~mask
    
    def comparison_selection(self, *items):
        ''' Calls the appropriate comparison method for each pair of operands and operators.'''
        logging.debug(f"comparison_selection() called with items: {repr(items)}")
        operands  = items[0::2]
        operators = items[1::2]
        operation = [getattr(self, token.type.lower(), None) for token in operators]
        mask = None
        for op, left, right, symbol in zip(operation, operands[:-1], operands[1:], operators):
            if op is None:
                raise ValueError(f"Unsupported operator: {symbol.type} '{symbol.value}'")
            part = op(left, right)
            mask = part if mask is None else mask & part
        return mask

    def number(self, token):
        logging.debug(f"number() called with token: {repr(token)}")
        v = token.value
        return float(v) if ('.' in v or 'e' in v or 'E' in v) else int(v)

    def const(self, token):
        logging.debug(f"const() called with token: {repr(token)}")
        return math.pi if token.lower() == 'pi' else math.e

    def func(self, fname, arg):
        logging.debug(f"func() called with fname: {repr(fname)}, arg: {repr(arg)}")
        name = fname.value.lower()
        if name == 'sq':
            return arg ** 2
        if name == 'abs':
            return np.abs(arg)
        return getattr(np, name)(arg)

    def var_sel(self, token):
        logging.debug(f"var_sel() called with token: {repr(token)}")
        return self.df[token.value[1:]]

    def selection_keyword(self, column) -> pd.Series:
        logging.debug(f"selection_keyword() called with obj: {repr(column)}")
        value = column.value
        if value == 'index':
            return self.df.index
        
        if value not in self.df.columns:
            raise ValueError(f"Column '{value}' not found in DataFrame.")
        return self.df[value]

    def property_selection(self, series: pd.Series, *values) -> pd.Series:
        logging.debug(f"property_selection() called with var: {repr(series)}, values: {repr(values)}")
        return series.isin(values)
    
    def range_branch(self, start, end, step=None):
        logging.debug(f"range_branch() called with start: {repr(start)}, end: {repr(end)}, step: {repr(step)}")
        return (start,end,step)

    def range_selection(self, series, range_selection):
        logging.debug(f"range_selection() called with series: {repr(series)}, branch: {repr(range_selection)}")
        start, end, step = range_selection
        sel = (series >= start) & (series <= end)
        if step is not None:
            sel &= (series - start) % step == 0
        return sel

    def regex_selection(self, series, _, pattern):
        logging.debug(f"regex_selection() called with series: {repr(series)}, pattern: {repr(pattern)}")
        pattern = pattern[1:-1] if pattern.startswith('"') else pattern
        return series.astype(str).str.contains(pattern, regex=True)

    def within_selection(self, token, dist, target):
        logging.debug(f"within_selection() called with token: {repr(token)}, dist: {dist}, target: {target}")
        ref_pts = self.df.loc[target, ['x', 'y', 'z']].values
        query_pts = self.df[['x', 'y', 'z']].values
        dist_matrix = np.sqrt(((query_pts[:, None, :] - ref_pts[None, :, :])**2).sum(axis=2))
        print(dir(token))
        if token.type == 'WITHIN':
            logging.debug(f"Calculating within distance: {dist}")
            return dist_matrix.min(axis=1) <= dist
        elif token.type == 'EXWITHIN':
            logging.debug(f"Calculating exwithin distance: {dist}")
            return dist_matrix.min(axis=1) > dist
        else:
            raise ValueError(f"Unsupported token type for within selection: {token.type}")


    def bonded_selection(self, items):
        logging.debug(f"bonded_selection() called with items: {repr(items)}")
        raise NotImplementedError("Bonded selection not implemented.")

    def sequence_selection_regex(self, _, pattern):
        logging.debug(f"sequence_selection_regex() called with pattern: {repr(pattern)}")
        pattern = pattern[1:-1] if pattern.startswith('"') else pattern
        return self.df['sequence'].str.contains(pattern, regex=True)

    def sequence_selection(self, _, pattern):
        logging.debug(f"sequence_selection() called with pattern: {repr(pattern)}")
        return self.df['sequence'].str.contains(pattern)

    def same_selection(self, _, col_token, __, mask):
        logging.debug(f"same_selection() called with col_token: {repr(col_token)}, mask: {repr(mask)}")
        col = col_token.value
        vals = self.df.loc[mask, col].unique()
        return self.df[col].isin(vals)

    def _expand_macro(self, expr, seen=None):
        """Recursively expand macro references in an expression string."""
        if seen is None:
            seen = set()
        logging.debug(f"Expanding macro: expr={expr!r}, seen={seen}")
        tokens = expr.split()
        expanded = []
        for token in tokens:
            macro_key = token[1:] if token.startswith('@') else token
            if macro_key in self.macros and macro_key not in seen:
                logging.debug(f"Expanding nested macro: {macro_key!r}")
                seen.add(macro_key)
                sub_expr = f"( {self.macros[macro_key]} )"
                expanded_sub = self._expand_macro(sub_expr, seen)
                expanded.append(expanded_sub)
                seen.remove(macro_key)
            else:
                logging.debug(f"Token is not a macro or already seen: {token!r}")
                expanded.append(token)
        result = ' '.join(expanded)
        logging.debug(f"Expanded macro result: {result!r}")
        return result

    def macro_sel(self, token):
        logging.debug(f"macro_sel() called with token: {repr(token)}")
        name = token.value.lstrip('@')
        expr = self.macros.get(name)
        if expr is None:
            raise ValueError(f"Macro {name!r} not defined")
        expanded = self._expand_macro(expr)
        subtree = self.parser.parse(expanded, start='expr')
        return self.transform(subtree)

    def bool_keyword(self, tok):
        logging.debug(f"bool_keyword() called with tok: {repr(tok)}")
        if tok.type == 'ALL':
            return self.select_all()
        if tok.type == 'NONE':
            return self.select_none()
        if tok.type == 'MACRO':
            return self.macro_sel(tok)
        return self.macro_sel(tok) if tok.value in self.macros else self.df[tok.value].astype(bool)

    def escaped_string(self, token):
        logging.debug(f"escaped_string() called with token: {repr(token)}")
        value = token.value.strip('"').replace('\\"', '"')
        return value

    def number(self, token):
        logging.debug(f"number() called with token: {repr(token)}")
        v = token.value
        return float(v) if ('.' in v or 'e' in v or 'E' in v) else int(v)
    
    def string_value(self, token):
        logging.debug(f"string_value() called with token: {repr(token)}")
        if isinstance(token, list):
            logging.debug(f"string_value() token is a list of length {len(token)} and type {type(token[0])}")
        return token.value
    

# Small test to ensure the transformer works
def test_transformer():
    from lark import exceptions
    logging.basicConfig(level=logging.DEBUG)
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'z': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'name': ['CA', 'CB', 'N', 'O', 'FE', 'CA', 'CB', 'O', 'N', 'H'],
        'mass': [12.0, 13.0, 14.0, 16.0, 55.8, 12.0, 13.0, 16.0, 14.0, 1.0],
        'resid': [1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
        'resname': ['ALA', 'ALA', 'GLY', 'GLY', 'HEM', 'PHE', 'PHE', 'HOH', 'HOH', 'GLU'],
        'element': ['C', 'C', 'N', 'O', 'Fe', 'C', 'C', 'O', 'N', 'H'],
    }

    EXAMPLES = [
    # --- Simple flags and terms ---
    # ("Simple flag: protein", "protein"),
    # ("Simple flag: water", "water"),
    # ("Simple field: name CA", "name CA"),
    # ("Multiple names", "name CA CB"),
    # ("Multiple resnames", "resname ALA GLY"),
    # ("Residue id", "resid 4"),
    # ("Index", "index 5"),
    # ("Backbone", "backbone"),
    # ("acidic", "acidic"),
    # ("All atoms", "all"),
    # ("None atoms", "none"),
    # ("Waters alias", "waters"),
    # ("Is_protein alias", "is_protein"),
    # ("Is_water alias", "is_water"),
    # ("Everything alias", "everything"),
    # ("Nothing alias", "nothing"),
    # ("Name with quotes 2", 'name "CA"'),
    # ("Name with quotes 3", 'name "CA" "CB" "CA CB"'),

    # --- Logic and default AND ---
    ("AND logic", "protein and water"),
    ("OR logic", "protein or water"),
    ("NOT logic", "not water"),
    ("Default AND", "not water acidic"),
    ("Default AND with fields", "resname ALA PHE name CA CB"),
    ("Default AND with flags", "acidic calpha"),
    ("Parentheses", "(protein or water) and not acidic"),
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
    ("Binary selection operator", "resid < 1"),
    ("Binary selection operator (ne)", "resid ne 0"),
    ("Binary selection operator (eq)", "resid eq 1"),
    ("Binary selection operator (ge)", "resid ge 1"),
    ("Binary selection operator (le)", "resid le 1"),
    ("Binary selection operator (gt)", "resid gt 1"),
    ("Binary selection operator (lt)", "resid lt 1"),
    ("Reverse binary selection", "1 == name"),
    ("Reverse binary selection (eq)", "1 eq name"),

    # --- Regular expressions ---
    ("Regex on resname", 'resname "A.*"'),
    #("Regex on sequence", 'sequence "MI.*DKQ"'), #TODO
    #("Regex on name", 'name =~ "C.*"'), #TODO
    #("Regex on name with AND", '(name =~ "C.*") and all'), #TODO

    # --- Distance-based selections ---
    ("Within distance", "within 5 of water"),
    ("Exwithin distance", "exwithin 3 of water"),
    ("Within distance of field", "within 5 of name FE"),
    ("Within with parentheses", "within 5 of (backbone or sidechain)"),

    # --- Sequence and macros --- #TODO
    # ("Sequence pattern", 'sequence "C..C"'),
    # ("Macro usage", "@alanine"),
    # ("User variable", "$var1 1"),
    # ("Macro with AND", "protein and @alanine"),
    # ("User variable complex", "within 8 of ($center 1 and @alanine)"),

    # --- Parentheses and complex logic ---
    ("Parentheses", "(protein or water) and not acidic"),
    ("Same residue as", "same resid as exwithin 4 of water"),
    ("Same resname as", "same resname as (protein within 6 of water)"),
    ("Complex logic", "protein or water or all"),
    ("Nested bool", "nothing and water or all"),
    ("Nested bool with parens", "nothing and (water or all)"),
    ("Quotes test 1", "name CA and resname ALA"),
    ("Quotes test 2", 'name "CA" and resname ALA'),
    ("In operator with resname", "resname ALA ASP GLU"),
    ("In operator with resid", "resid 100 101 102"),

    # ---Complex selections ---
    ("Complex selection", "protein and (resname ALA or resname GLY) and not water"),
    ("Complex selection with parentheses", "(protein or water) and not acidic"),
    # ("Complex selection with regex", 'resname "A.*" and name =~ "C.*"'), #TODO
    ("Complex selection with distance", "protein within 5 of (resname ALA or resname GLY)"),
    # ("Complex selection with sequence", 'sequence "C..C" and resname ALA'), #TODO
    # ("Complex selection with macro", "@alanine and within 5 of water"), #TODO
    # ("Complex selection with user variable", "$var1==1 and 0<$var2<2"), #TODO
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
    ("Complex comparison", "(1*1+1-1) < (1*x+2-2) < (3//3*3)"),

    # --- Wrong syntax examples ---
    ("Wrong syntax: missing operator", "101 102 103"),
    ]


    # 1) build the parser (no transformer yet)
    grammar_path = os.path.join(os.path.dirname(__file__), "../utils/selection_syntax.lark")
    with open(grammar_path) as f:
        grammar_text = f.read()
    base_parser = Lark(grammar_text, parser='lalr', propagate_positions=True, debug =True, start=['start', 'expr'])
    


    df = pd.DataFrame(data)
    transformer = PandasTransformer(df, parser=base_parser)

    
    # with open("molscene/utils/selection_syntax.lark", "r") as f:
    #     parser = Lark(f.read(), parser='lalr', transformer=PandasTransformer(df), debug=True)

    for example in EXAMPLES:
        description, sel = example
        print(f"Testing: {description}")
        try:
            tree  = base_parser.parse(sel, start='start')
            print("Parse tree:")
            print(tree.pretty())
        except exceptions.LarkError as e:
            print(f"Parse failed: {e.__class__.__name__}: {e}")

        # Test a simple selection
        result = transformer.transform(tree)
        assert isinstance(result, pd.DataFrame)
        print("Selection result:")
        print(result)

if __name__ == "__main__":
    test_transformer()
    print("Transformer tests passed.")