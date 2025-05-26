import numpy as np
import math
import pandas as pd
from lark import Lark, Transformer, v_args
import logging

@v_args(inline=True)
class PandasTransformer(Transformer):
    from operator import add, sub, mul, truediv, floordiv, mod, pow, neg, and_, or_, xor, lt, gt, le, ge, eq, ne
    
    def __init__(self, df, macros=None):
        super().__init__()
        self.df = df
        self.macros = macros or {}
        logging.debug(f"PandasTransformer initialized with DataFrame of shape {df.shape} and macros: {list(self.macros.keys())}")

    def start(self, mask):
        logging.debug(f"start() called with mask of type {type(mask).__name__}: {repr(mask)}")
        return self.df[mask]

    def not_(self, mask):
        logging.debug(f"not_() called with mask: {repr(mask)}")
        return ~mask[0]
    
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

    def selection_keyword(self, obj):
        logging.debug(f"selection_keyword() called with obj: {repr(obj)}")
        if isinstance(obj, pd.Series):
            return obj
        return self.df[obj.value]

    def property_selection(self, var, *values):
        logging.debug(f"property_selection() called with var: {repr(var)}, values: {repr(values)}")
        return self.df[var].isin(values)
    
    def range_branch(self, items):
        logging.debug(f"range_branch() called with items: {repr(items)}")
        return items

    def range_selection(self, series, branch):
        logging.debug(f"range_selection() called with series: {repr(series)}, branch: {repr(branch)}")
        if len(branch) == 2:
            low, high = branch
            return (series >= low) & (series <= high)
        start, end, step = branch
        return ((series >= start) & (series <= end) & ((series - start) % step == 0))

    def regex_selection(self, series, _, pattern):
        logging.debug(f"regex_selection() called with series: {repr(series)}, pattern: {repr(pattern)}")
        pattern = pattern[1:-1] if pattern.startswith('"') else pattern
        return series.astype(str).str.contains(pattern, regex=True)

    def within_selection(self, _, dist, target):
        logging.debug(f"within_selection() called with dist: {repr(dist)}, target: {repr(target)}")
        ref_pts = self.df.loc[target, ['x', 'y', 'z']].values
        query_pts = self.df[['x', 'y', 'z']].values
        if len(ref_pts) == 0:
            return pd.Series(False, index=self.df.index)
        dist_matrix = np.sqrt(((query_pts[:, None, :] - ref_pts[None, :, :])**2).sum(axis=2))
        return dist_matrix.min(axis=1) <= dist

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

    def macro_sel(self, token):
        logging.debug(f"macro_sel() called with token: {repr(token)}")
        name = token.value[1:]
        expr = self.macros.get(name)
        if expr is None:
            raise ValueError(f"Macro '{name}' not defined.")
        from lark import Lark
        parser = Lark.open("grammar.lark", start="start")
        tree = parser.parse(expr)
        return PandasTransformer(self.df, self.macros).transform(tree)

    def bool_keyword(self, tok):
        logging.debug(f"bool_keyword() called with tok: {repr(tok)}")
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
        'x': [1, 2, 3],
        'y': [4, 5, 6],
        'z': [7, 8, 9],
        'name': ['A', 'B', 'C'],
        'mass': [12.0, 13.0, 14.0]
    }
    df = pd.DataFrame(data)
    with open("molscene/utils/selection_syntax.lark", "r") as f:
        parser = Lark(f.read(), parser='lalr', transformer=PandasTransformer(df), debug=True)

    sel = '1**2 < x**2 < 3**2'    

    try:
        with open("molscene/utils/selection_syntax.lark", "r") as f:
            parser_tree = Lark(f.read(), parser='lalr', debug=True)
        tree = parser_tree.parse(sel)
        print("Parse tree:")
        print(tree.pretty())
    except exceptions.LarkError as e:
        print(f"Parse failed: {e.__class__.__name__}: {e}")

    # Test a simple selection
    result = parser.parse(sel)
    assert isinstance(result, pd.DataFrame)
    print("Selection result:")
    print(result)

if __name__ == "__main__":
    test_transformer()
    print("Transformer tests passed.")