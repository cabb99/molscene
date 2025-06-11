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
        # Evaluate right only on where left_mask is False
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
    field: Node  # always a Node now
    op: str
    value: Union[str, Node, float, int, None]
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        left = self.field.evaluate(df)
        right = self.value.evaluate(df) if isinstance(self.value, Node) else self.value

        op = self.op
        if not isinstance(left, pd.Series) and isinstance(right, pd.Series):
            left, right = right, left
            flip = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!=',
                    'eq': 'eq', 'ne': 'ne', 'lt': 'gt', 'gt': 'lt', 'le': 'ge', 'ge': 'le'}
            op = flip.get(op, op)

        if right is None:
            return left.astype(bool)

        ops = {
            '==': lambda l, r: l == r,
            '!=': lambda l, r: l != r,
            '<':  lambda l, r: l < r,
            '>':  lambda l, r: l > r,
            '<=': lambda l, r: l <= r,
            '>=': lambda l, r: l >= r,
            'eq': lambda l, r: l == r,
            'ne': lambda l, r: l != r,
            'lt': lambda l, r: l < r,
            'gt': lambda l, r: l > r,
            'le': lambda l, r: l <= r,
            'ge': lambda l, r: l >= r,
        }
        try:
            return ops[op](left, right)
        except TypeError:
            # Return all False if comparison is invalid (e.g., str < int)
            if isinstance(left, pd.Series):
                return pd.Series(False, index=left.index)
            elif isinstance(right, pd.Series):
                return pd.Series(False, index=right.index)
            else:
                return False
    def symbolic(self) -> str:
        return f"{self.field.symbolic()} {self.op} {self.value.symbolic() if isinstance(self.value, Node) else self.value}"

@dataclass
class Range(Node):
    start: Union[int, float, Node]
    end: Union[int, float, Node]
    step: Union[int, float, Node, None] = None
    def evaluate(self, df: DataFrameLike):
        raise NotImplementedError("Range should be evaluated in PropertySelection, not directly.")
    def symbolic(self) -> str:
        expr = f"Range({self.start!r}, {self.end!r}"
        if self.step is not None:
            expr += f", {self.step!r}"
        expr += ")"
        return expr

@dataclass
class Regex(Node):
    field: Node  # always a Node now
    pattern: str
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        col = self.field.evaluate(df)
        return col.astype(str).str.contains(self.pattern, regex=True)
    def symbolic(self) -> str:
        return f"{self.field.symbolic()}.astype(str).str.contains({self.pattern!r})"

@dataclass
class Within(Node):
    """Spatial selection within a distance of reference points."""
    distance: Node  # always a Node now
    target_mask: Node  # always a Node now
    mode: str = "within"  # "within" or "exwithin"
    def evaluate(self, df: DataFrameLike) -> pd.Series:
        distance = self.distance.evaluate(df)
        mask = self.target_mask.evaluate(df)
        ref_pts = df.loc[mask, ['x','y','z']].values
        pts = df[['x','y','z']].values
        if ref_pts.size == 0:
            # “within” of an empty set → nobody matches
            result = np.zeros(len(df), dtype=bool)
            return pd.Series(result, index=df.index)
        d2 = ((pts[:,None,:] - ref_pts[None,:,:])**2).sum(axis=2)
        if self.mode == "within":
            result = (d2.min(axis=1)**0.5) <= distance
        else:  # exwithin
            result = (d2.min(axis=1)**0.5) > distance
        return pd.Series(result, index=df.index)
    def symbolic(self) -> str:
        return f"{self.mode}({self.distance.symbolic()}, {self.target_mask.symbolic()})"

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
    field: Node  # always a Node now
    values: list
    def evaluate(self, df):
        col = self.field.evaluate(df)
        mask = pd.Series(False, index=col.index)
        for v in self.values:
            if isinstance(v, Range):
                start = v.start.evaluate(df) if isinstance(v.start, Node) else v.start
                end = v.end.evaluate(df) if isinstance(v.end, Node) else v.end
                step = v.step.evaluate(df) if (v.step is not None and isinstance(v.step, Node)) else v.step
                range_mask = (col >= start) & (col <= end)
                if step is not None:
                    range_mask &= ((col - start) % step == 0)
                mask |= range_mask
            else:
                mask |= (col == v.evaluate(df) if isinstance(v, Node) else col == v)
        return mask
    def symbolic(self):
        return f"PropertySelection({self.field.symbolic()}, {self.values})"

@dataclass
class Same(Node):
    field: Node  # always a Node now
    mask: Node
    def evaluate(self, df):
        col = self.field.evaluate(df)
        vals = col[df.index[self.mask.evaluate(df)]].unique()
        return col.isin(vals)
    def symbolic(self):
        return f"Same({self.field.symbolic()}, {self.mask.symbolic()})"

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

@dataclass
class Bonded(Node):
    distance: float
    selection: Node
    def evaluate(self, df):
        raise NotImplementedError("Bonded selection not implemented.")
    def symbolic(self):
        return f"Bonded({self.distance!r}, {self.selection!r})"

@dataclass
class SequenceSelectionRegex(Node):
    pattern: str
    def evaluate(self, df):
        raise NotImplementedError("Sequence selection regex not implemented.")
    def symbolic(self):
        return f"SequenceSelectionRegex({self.pattern!r})"

@dataclass
class SequenceSelection(Node):
    sequence: str
    def evaluate(self, df):
        raise NotImplementedError("Sequence selection not implemented.")
    def symbolic(self):
        return f"SequenceSelection({self.sequence!r})"

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
        # Pass the raw operator text
        return Comparison(left, op_tok.value, right)

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

    def range_selection(self, name, start, end, step=None):
        start = self._to_node(start)
        end = self._to_node(end)
        if step is not None:
            step = self._to_node(step)

        return Range(name, start, end, step)

    def regex(self, name, pattern_tok):
        # name_tok may be Tree, so transform
        name = self._to_node(name)
        if isinstance(name, SelectionKeyword):
            name = name.name
        return Regex(name, pattern_tok.value[1:-1])

    def within_selection(self, within_token, dist, target_mask):
        mode = str(within_token).lower()
        return Within(dist, self._to_node(target_mask), mode=mode)

    def property_selection(self, name, *values):
        # Recursively transform all values, including handling Tree nodes for ranges
        vals = []
        name = self._to_node(name)
        for v in values:
            node = self._to_node(v)
            vals.append(node)
        return PropertySelection(name, vals)

    def same_selection(self, name, mask):
        name = self._to_node(name)
        return Same(name, self._to_node(mask))

    def macro(self, name):
        # Macro expansion will be handled in Evaluator
        return Macro(str(name).lstrip('@'), None)

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

    def xor_(self, left, right):
        return Xor(self._to_node(left), self._to_node(right))

    def comparison_selection(self, *items):
        operands = [self._to_node(x) for x in items[0::2]]
        operators = items[1::2]
        mask = None
        for left, op, right in zip(operands, operators, operands[1:]):
            cmp = Comparison(left, str(op), right)
            mask = cmp if mask is None else And(mask, cmp)
        return mask

    def bonded_selection(self, distance, selection):
        return Bonded(distance, self._to_node(selection))

    def sequence_selection_regex(self, pattern):
        return SequenceSelectionRegex(str(pattern))

    def sequence_selection(self, sequence):
        return SequenceSelection(str(sequence))

    def escaped_string(self, tok):
        v = tok.value
        return v[1:-1].replace('\\"', '"') if v.startswith('"') else v

    def single_quoted_string(self, tok):
        v = tok.value
        return v[1:-1].replace("\\'", "'") if v.startswith("'") else v

    def var_sel(self, tok):
        return SelectionKeyword(str(tok))

    def macro_sel(self, tok):
        return Macro(str(tok).lstrip('@'), None)

    def selection_keyword(self, token):
        # If token is a Tree, transform it
        if isinstance(token, Tree):
            return self._to_node(token)
        return SelectionKeyword(str(token))

    def regex_selection(self, operand, pattern):
        # operand may be a Tree or Token; pattern is a Token
        name = self._to_node(operand)
        if isinstance(name, SelectionKeyword):
            name = name.name
        return Regex(str(name), pattern.value[1:-1])

    def start(self, expr):
        return self._to_node(expr)

    def number_range_selection(self, start, end, step=None):
        if step is not None:
            return Range(start, end, step)
        return Range(start, end)
    
    def stride_range_selection(self, start, end, step=None):
        if step is not None:
            return Range(start, end, step)
        return Range(start, end)

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