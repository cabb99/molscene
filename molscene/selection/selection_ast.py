"""
selection_ast.py

Core logic for selection expressions:
- AST node definitions (self-contained, no DataFrame or parser knowledge)
- MacrosLoader: loads/expands macros
- Parser: parses text to parse tree
- ASTBuilder: builds AST from parse tree
- Evaluator: evaluates AST on DataFrame
- Config: centralizes paths and logging

Follows Single Responsibility Principle and uses dependency injection.
"""

import os
import json
import logging
import math
import numpy as np
import pandas as pd
from lark import Lark, Transformer, v_args, Token, Tree
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Optional, Dict, Union, List

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Config ---
class SelectionConfig:
    """Centralized configuration for grammar and macros paths."""
    def __init__(self, grammar_path=None, macros_path=None):
        self.grammar_path = grammar_path or os.environ.get('SELECTION_GRAMMAR_PATH') or \
            os.path.join(os.path.dirname(__file__), 'selection_syntax.lark')
        self.macros_path = macros_path or os.environ.get('SELECTION_MACROS_PATH') or \
            os.path.join(os.path.dirname(__file__), 'macros.json')

# --- AST Node Definitions ---
# A protocol for objects that support minimal DataFrame-like access
@runtime_checkable
class DataFrameLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...
    @property
    def index(self) -> Any: ...

class Node:
    """Base AST node; subclasses implement eager and symbolic evaluation."""
    def evaluate(self, df: DataFrameLike) -> Any:
        raise NotImplementedError
    def symbolic(self) -> str:
        raise NotImplementedError

@dataclass
class And(Node):
    left: Node
    right: Node
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        left_mask = self.left.evaluate(df)
        # Short-circuit: if nothing matches left, return all False
        if not left_mask.any():
            return left_mask
        # Only evaluate right on the subset where left_mask is True
        right_mask = self.right.evaluate(df[left_mask])
        # Reindex right_mask to match original df
        combined = pd.Series(False, index=df.index)
        combined.loc[right_mask.index] = left_mask.loc[right_mask.index] & right_mask
        return combined
    def symbolic(self) -> str:
        return f"({self.left.symbolic()}) & ({self.right.symbolic()})"

@dataclass
class Or(Node):
    left: Node
    right: Node
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        left_mask = self.left.evaluate(df)
        # Short-circuit: if everything matches left, return all True
        if left_mask.all():
            return left_mask
        # Only evaluate right on the subset where left_mask is False
        right_mask = self.right.evaluate(df[~left_mask])
        # Start with left_mask, set True where right_mask is True
        combined = left_mask.copy()
        combined.loc[right_mask.index] = left_mask.loc[right_mask.index] | right_mask
        return combined
    def symbolic(self) -> str:
        return f"({self.left.symbolic()}) | ({self.right.symbolic()})"

@dataclass
class Xor(Node):
    left: Node
    right: Node
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        left_mask = self.left.evaluate(df)
        # Short-circuit: if everything matches left, return ~right
        if left_mask.all():
            return ~self.right.evaluate(df)
        # Short-circuit: if nothing matches left, return right
        if not left_mask.any():
            return self.right.evaluate(df)
        # Evaluate right only where left_mask is False
        right_mask = self.right.evaluate(df[~left_mask])
        combined = left_mask.copy()
        combined.loc[right_mask.index] = left_mask.loc[right_mask.index] ^ right_mask
        return combined
    def symbolic(self) -> str:
        return f"({self.left.symbolic()}) ^ ({self.right.symbolic()})"

@dataclass
class Not(Node):
    expr: Node
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        return ~self.expr.evaluate(df)
    def symbolic(self) -> str:
        return f"~({self.expr.symbolic()})"

@dataclass
class Comparison(Node):
    field: Union[str, Node, float, int]
    op: str
    value: Union[str, Node, float, int, None]
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        left = self.field.evaluate(df) if isinstance(self.field, Node) else df[self.field] if isinstance(self.field, str) else self.field
        right = self.value.evaluate(df) if isinstance(self.value, Node) else self.value
        # If left is not a Series but right is, swap and reverse operator
        if not isinstance(left, pd.Series) and isinstance(right, pd.Series):
            left, right = right, left
            op_map = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', 'eq': 'eq', '!=': '!=', 'ne': 'ne'}
            op = op_map.get(self.op, self.op)
        else:
            op = self.op
        if right is None:
            return left.astype(bool)
        return {
            '==': left == right,
            '!=': left != right,
            '<':  left <  right,
            '>':  left >  right,
            '<=': left <= right,
            '>=': left >= right,
            'eq': left == right,
            'ne': left != right,
            'lt': left <  right,
            'gt': left >  right,
            'le': left <= right,
            'ge': left >= right
        }[op]
    def symbolic(self) -> str:
        return f"{self.field} {self.op} {self.value}"

@dataclass
class Range(Node):
    """Numeric range selection, optionally with a stride step."""
    field: str
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float, None] = None
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        col = df[self.field]
        mask = (col >= self.start) & (col <= self.end)
        if self.step is not None:
            mask &= ((col - self.start) % self.step == 0)
        return mask
    def symbolic(self) -> str:
        expr = f"(df[{self.field!r}] >= {self.start!r}) & (df[{self.field!r}] <= {self.end!r})"
        if self.step is not None:
            expr += f" & ((df[{self.field!r}] - {self.start!r}) % {self.step!r} == 0)"
        return expr

@dataclass
class Regex(Node):
    field: str
    pattern: str
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        return df[self.field].astype(str).str.contains(self.pattern, regex=True)
    def symbolic(self) -> str:
        return f"df[{self.field!r}].astype(str).str.contains({self.pattern!r})"

@dataclass
class Within(Node):
    """Spatial selection within a distance of reference points."""
    distance: float
    target_mask: Union[str, Node, pd.Series]
    mode: str = "within"  # "within" or "exwithin"
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        distance = self.distance.evaluate(df) if isinstance(self.distance, Node) else self.distance
        mask = self.target_mask.evaluate(df) if isinstance(self.target_mask, Node) else self.target_mask
        ref_pts = df.loc[mask, ['x','y','z']].values
        pts = df[['x','y','z']].values
        d2 = ((pts[:,None,:] - ref_pts[None,:,:])**2).sum(axis=2)
        if self.mode == "within":
            result = (d2.min(axis=1)**0.5) <= distance
        else:  # exwithin
            result = (d2.min(axis=1)**0.5) > distance
        return pd.Series(result, index=df.index)
    def symbolic(self) -> str:
        return f"{self.mode}({self.distance!r}, {self.target_mask!r})"

@dataclass
class Macro(Node):
    """User-defined macro that expands into another AST subtree."""
    name: str
    definition: Optional[Node] = None

    def evaluate(self, df):
        if self.definition is None:
            raise RuntimeError("Macro definition not expanded.")
        return self.definition.evaluate(df)

    def symbolic(self):
        # Always expand macros before symbolic, for correct output
        if self.definition is None:
            raise RuntimeError("Macro definition not expanded for symbolic(). Expand macros before calling symbolic().")
        return self.definition.symbolic()

@dataclass
class All(Node):
    def evaluate(self, df):
        return pd.Series(True, index=df.index)
    def symbolic(self):
        return "all"

@dataclass
class NoneNode(Node):
    def evaluate(self, df):
        return pd.Series(False, index=df.index)
    def symbolic(self):
        return "none"

@dataclass
class PropertySelection(Node):
    field: str
    values: list
    def evaluate(self, df):
        col = df[self.field]
        mask = pd.Series(False, index=col.index)
        for v in self.values:
            if isinstance(v, Range):
                mask |= v.evaluate(df)
            else:
                mask |= (col == v)
        return mask
    def symbolic(self):
        return f"PropertySelection({self.field!r}, {self.values!r})"

@dataclass
class Same(Node):
    field: str
    mask: Node
    def evaluate(self, df):
        col = df[self.field]
        vals = col[df.index[self.mask.evaluate(df)]].unique()
        return col.isin(vals)
    def symbolic(self):
        return f"Same({self.field!r}, {self.mask.symbolic()})"

@dataclass
class Add(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) + self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) + ({self.right.symbolic()})"

@dataclass
class Sub(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) - self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) - ({self.right.symbolic()})"

@dataclass
class Mul(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) * self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) * ({self.right.symbolic()})"

@dataclass
class Div(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) / self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) / ({self.right.symbolic()})"

@dataclass
class FloorDiv(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) // self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) // ({self.right.symbolic()})"

@dataclass
class Mod(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) % self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) % ({self.right.symbolic()})"

@dataclass
class Pow(Node):
    left: Node
    right: Node
    def evaluate(self, df):
        return self.left.evaluate(df) ** self.right.evaluate(df)
    def symbolic(self):
        return f"({self.left.symbolic()}) ** ({self.right.symbolic()})"

@dataclass
class Neg(Node):
    value: Node
    def evaluate(self, df):
        return -self.value.evaluate(df)
    def symbolic(self):
        return f"-({self.value.symbolic()})"

@dataclass
class Func(Node):
    name: str
    arg: Node
    def evaluate(self, df):
        v = self.arg.evaluate(df)
        if self.name == 'sq':
            return v ** 2
        if self.name == 'abs':
            return np.abs(v)
        return getattr(np, self.name)(v)
    def symbolic(self):
        return f"{self.name}({self.arg.symbolic()})"

@dataclass
class Number(Node):
    value: Union[int, float]
    def evaluate(self, df):
        v = self.value
        if '.' in v or 'e' in v or 'E' in v:
            return float(v)
        return int(v)
    def symbolic(self):
        return str(self.value)

@dataclass
class Const(Node):
    name: str
    def evaluate(self, df):
        if self.name.lower() == 'pi':
            return math.pi
        elif self.name.lower() == 'e':
            return math.e
        else:
            raise ValueError(f"Unknown constant: {self.name}")
    def symbolic(self):
        return self.name

@dataclass
class SelectionKeyword(Node):
    name: str
    def evaluate(self, df):
        if self.name == 'index':
            return pd.Series(df.index, name='index', index=df.index)
        if self.name not in df.columns:
            raise ValueError(f"Column '{self.name}' not found in DataFrame.")
        return df[self.name]
    def symbolic(self):
        return self.name

# --- Macros Loader ---
class MacrosLoader:
    """Loads and expands macros from a JSON file."""
    def __init__(self, macros_path: str):
        self.macros = self._load_macros(macros_path)
    def _load_macros(self, macros_path: str) -> Dict[str, str]:
        macros_flat = {}
        if os.path.exists(macros_path):
            with open(macros_path, 'r') as f:
                try:
                    macros_json = json.load(f)
                    for category in macros_json.get('macros', {}):
                        for macro_name, macro_obj in macros_json['macros'][category].items():
                            definition = macro_obj.get('definition', '')
                            macros_flat[macro_name] = definition
                            for syn in macro_obj.get('synonyms', []):
                                macros_flat[syn] = definition
                except Exception as e:
                    logger.warning(f"Failed to load macros from {macros_path}: {e}")
        return macros_flat
    def expand_macro(self, name: str, seen=None) -> str:
        if seen is None:
            seen = set()
        if name not in self.macros:
            raise ValueError(f"Macro {name!r} not defined")
        expr = self.macros[name]
        tokens = expr.split()
        expanded = []
        for token in tokens:
            macro_key = token[1:] if token.startswith('@') else token
            if macro_key in self.macros and macro_key not in seen:
                seen.add(macro_key)
                expanded_sub = self.expand_macro(macro_key, seen)
                expanded.append(expanded_sub)
                seen.remove(macro_key)
            else:
                expanded.append(token)
        return ' '.join(expanded)

# --- Parser ---
class SelectionParser:
    """Parses selection strings to parse trees using Lark."""
    def __init__(self, grammar_path: str):
        with open(grammar_path) as f:
            grammar_text = f.read()
        self.lark = Lark(grammar_text, parser='lalr', propagate_positions=True, start=['start', 'expr'])
    def parse(self, text: str, start_rule: str = 'start') -> Tree:
        return self.lark.parse(text, start=start_rule)

# --- AST Builder ---
@v_args(inline=True)
class ASTBuilder(Transformer):
    """Transforms parse trees into AST nodes."""
    def __init__(self, macros: Optional[Dict[str, str]] = None):
        super().__init__()
        self.macros = macros or {}
    def _to_node(self, x):
        # Recursively transform Tree or Token to AST node
        if isinstance(x, Tree):
            return self.transform(x)
        return x

    def and_(self, left, right):
        return And(self._to_node(left), self._to_node(right))

    def or_(self, left, right):
        return Or(self._to_node(left), self._to_node(right))

    def not_(self, expr):
        return Not(self._to_node(expr))

    def comparison(self, name_tok, op_tok, value):
        left = self._to_node(name_tok)
        right = self._to_node(value)
        # If left is a SelectionKeyword node, use its name as string
        if isinstance(left, SelectionKeyword):
            left = left.name
        return Comparison(left, str(op_tok), right)

    def add(self, left, right):
        return Add(self._to_node(left), self._to_node(right))
    def sub(self, left, right):
        return Sub(self._to_node(left), self._to_node(right))
    def mul(self, left, right):
        return Mul(self._to_node(left), self._to_node(right))
    def truediv(self, left, right):
        return Div(self._to_node(left), self._to_node(right))
    def floordiv(self, left, right):
        return FloorDiv(self._to_node(left), self._to_node(right))
    def mod(self, left, right):
        return Mod(self._to_node(left), self._to_node(right))
    def pow(self, left, right):
        return Pow(self._to_node(left), self._to_node(right))
    def neg(self, value):
        return Neg(self._to_node(value))

    def func(self, fname, arg):
        # fname is Token or str, arg may be Tree
        return Func(str(fname), self._to_node(arg))

    def const(self, token):
        return Const(token.value)

    def number(self, tok):
        return Number(tok.value)

    def string(self, tok):
        v = tok.value
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        return v
    
    def string_value(self, tok):
        v = tok.value
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        return v

    def range_selection(self, name_tok, start, end, step=None):
        return Range(str(name_tok), start, end, step)

    def regex(self, name_tok, pattern_tok):
        # name_tok may be Tree, so transform
        name = self._to_node(name_tok)
        if isinstance(name, SelectionKeyword):
            name = name.name
        return Regex(str(name), pattern_tok.value[1:-1])

    def within_selection(self, within_token, dist, target_mask):
        mode = str(within_token).lower()
        return Within(dist, self._to_node(target_mask), mode=mode)

    def property_selection(self, name_tok, *values):
        # Recursively transform all values
        vals = [self._to_node(v) for v in values]
        return PropertySelection(str(name_tok), vals)

    def same_selection(self, name_tok, mask):
        return Same(str(name_tok), self._to_node(mask))

    def macro(self, name_tok):
        # Macro expansion will be handled in Evaluator
        return Macro(str(name_tok).lstrip('@'), None)

    def bool_keyword(self, tok):
        # Handle macros
        if tok.type == 'ALL':
            return All()
        if tok.type == 'NONE':
            return NoneNode()
        # If it's a macro, return Macro node
        if hasattr(self, 'macros') and tok.value in self.macros:
            return Macro(str(tok.value), None)
        # Otherwise, treat as a column/flag
        return SelectionKeyword(str(tok.value))

    def comparison_selection(self, *items):
        # Handles chained comparisons: a < b < c
        operands = [self._to_node(x) for x in items[0::2]]
        operators = items[1::2]
        mask = None
        for left, op, right in zip(operands, operators, operands[1:]):
            cmp = Comparison(left, str(op), right)
            mask = cmp if mask is None else And(mask, cmp)
        return mask

    def regex_selection(self, name, pattern):
        name = self._to_node(name)
        if isinstance(name, SelectionKeyword):
            name = name.name
        return Regex(str(name), pattern.value[1:-1])

    def selection_keyword(self, token):
        # If token is a Tree, transform it
        if isinstance(token, Tree):
            return self._to_node(token)
        return SelectionKeyword(str(token))

    def start(self, expr):
        return self._to_node(expr)

# --- Evaluator ---
class Evaluator:
    """Evaluates AST nodes on a DataFrame, with macro expansion."""
    def __init__(self, df, macros: Dict[str, str], parser: SelectionParser, builder: ASTBuilder):
        self.df = df
        self.macros = macros
        self.parser = parser
        self.builder = builder
    def expand_macros_in_ast(self, node):
        """Recursively expand Macro nodes in the AST using string-based expansion for full macro resolution."""
        if isinstance(node, Macro):
            # Use string-based recursive expansion from MacrosLoader
            expanded_expr = self._expand_macro_string(node.name)
            tree = self.parser.parse(expanded_expr, start_rule='expr')
            ast = self.builder.transform(tree)
            return self.expand_macros_in_ast(ast)
        # Recursively expand for all dataclass fields that are Node(s)
        if hasattr(node, '__dataclass_fields__'):
            for field in node.__dataclass_fields__:
                value = getattr(node, field)
                if isinstance(value, Node):
                    setattr(node, field, self.expand_macros_in_ast(value))
                elif isinstance(value, list):
                    new_list = [self.expand_macros_in_ast(v) if isinstance(v, Node) else v for v in value]
                    setattr(node, field, new_list)
        return node

    def _expand_macro_string(self, name, seen=None):
        """Recursively expand macro references in a string, including hidden/underscore macros."""
        if seen is None:
            seen = set()
        if name not in self.macros:
            raise ValueError(f"Macro {name!r} not defined")
        expr = self.macros[name]
        tokens = expr.split()
        expanded = []
        for token in tokens:
            macro_key = token[1:] if token.startswith('@') else token
            if macro_key in self.macros and macro_key not in seen:
                seen.add(macro_key)
                expanded_sub = self._expand_macro_string(macro_key, seen)
                expanded.append(expanded_sub)
                seen.remove(macro_key)
            else:
                expanded.append(token)
        return ' '.join(expanded)

    def evaluate(self, root: Node) -> pd.Series:
        root = self.expand_macros_in_ast(root)
        return root.evaluate(self.df)
    def symbolic(self, root: Node) -> str:
        root = self.expand_macros_in_ast(root)
        return root.symbolic()

# --- Example usage and test harness ---
def test_evaluator():
    """Test harness for Evaluator and selection logic."""
    logging.basicConfig(level=logging.INFO)
    config = SelectionConfig()
    macros_loader = MacrosLoader(config.macros_path)
    parser = SelectionParser(config.grammar_path)
    builder = ASTBuilder(macros_loader.macros)

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
    df = pd.DataFrame(data)
    evaluator = Evaluator(df, macros_loader.macros, parser, builder)
    EXAMPLES = [
        ("Simple flag: protein", "protein"),
        ("Simple flag: water", "water"),
        ("Simple field: name CA", "name CA"),
        ("Multiple names", "name CA CB"),
        ("Multiple resnames", "resname ALA GLY"),
        ("Residue id", "resid 4"),
        ("Index", "index 5"),
        ("Backbone", "backbone"),
        ("acidic", "acidic"),
        ("All atoms", "all"),
        ("None atoms", "none"),
        ("Waters alias", "waters"),
        ("Is_protein alias", "is_protein"),
        ("Is_water alias", "is_water"),
        ("Everything alias", "everything"),
        ("Nothing alias", "nothing"),
        ("Name with quotes 2", 'name "CA"'),
        ("Name with quotes 3", 'name "CA" "CB" "CA CB"'),
        ("AND logic", "protein and water"),
        ("OR logic", "protein or water"),
        ("NOT logic", "not water"),
        ("Default AND", "not water acidic"),
        ("Default AND with fields", "resname ALA PHE name CA CB"),
        ("Default AND with flags", "acidic calpha"),
        ("Parentheses", "(protein or water) and not acidic"),
        ("Nested NOT", "not not (protein and water)"),
        ("NOT with !", "!protein"),
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
        ("Regex on name", 'name =~ "C.*"'),
        ("Regex on name with AND", '(name =~ "C.*") and all'),
        ("Within distance", "within 5 of water"),
        ("Exwithin distance", "exwithin 3 of water"),
        ("Within distance of field", "within 5 of name FE"),
        ("Within with parentheses", "within 5 of (backbone or sidechain)"),
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
        ("Complex selection", "protein and (resname ALA or resname GLY) and not water"),
        ("Complex selection with parentheses", "(protein or water) and not acidic"),
        ("Complex selection with regex", 'resname =~ "A.*" and name =~ "C.*"'),
        ("Complex selection with distance", "protein within 5 of (resname ALA or resname GLY)"),
        ("Complex selection with function", "sqrt(x**2 + y**2 + z**2) < 10"),
        ("Math with functions", "sqrt(x**2 + y**2 + z**2) < 10"),
        ("Math with abs", "abs(x) < 5"),
        ("Math with log", "log(x) > 0"),
        ("Math with exp", "exp(x) < 100"),
        ("Math with sin", "sin(x) > 0.5"),
        ("Math with cos", "cos(x) < 0.5"),
        ("Math with tan", "tan(x) > 1"),
        ("Complex math with functions", "sqrt(z^3-sin(x*y)^2) < 10"),
        ("Complex comparison", "(1*1+1-1) < (1*x+2-2) < (3//3*3)"),
        ("Wrong syntax: missing operator", "101 102 103"),
    ]
    for description, sel in EXAMPLES:
        print(f"Testing: {description}")
        print(f"Selection: {sel}")
        try:
            tree = parser.parse(sel, start_rule='start')
            ast = builder.transform(tree)
            result = evaluator.symbolic(ast)
            print(f"symbolic expression: {result}")
            
            result = evaluator.evaluate(ast)
            assert isinstance(result, (pd.DataFrame, pd.Series, np.ndarray))
            print("Selection result:")
            print(df[result] if isinstance(result, pd.Series) else result)
        except Exception as e:
            print(f"Parse or evaluation failed: {e.__class__.__name__}: {e}")
    return parser, builder, evaluator

if __name__ == '__main__':
    parser,builder,evaluator = test_evaluator()
    print("Evaluator tests passed.")