import re
from pathlib import Path

# 1) Read the entire grammar
grammar = Path("molscene/selection/selection_syntax.lark").read_text()

# 2) Pull out only the literal keywords (those defined as "…")
#    e.g. NAME_KW  : "name"        -> we capture "name"
# reserved = re.findall(r'^[A-Z_][A-Z0-9_\.]*\s*:\s*"([^"]+)"', grammar, flags=re.MULTILINE)
# 2) Pull out *every* quoted literal in the grammar,
#     then filter out purely-punctuation ones if you like.
# all_literals = re.findall(r'"([^"]+)"', grammar)
# # (optional) only keep things that look like words or numbers
# reserved = [lit for lit in all_literals if re.match(r'^[A-Za-z0-9_]+$', lit)]

grammar_no_comments = re.sub(r'//.*', '', grammar)
reserved = []
for line in grammar_no_comments.splitlines():
    # only lines that look like UPPERCASE_TOKEN : "…"
    m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*:\s*(.+)$', line)
    if not m:
        continue
    rhs = m.group(2)
    # find all "…"
    for lit in re.findall(r'"([^"]+)"', rhs):
        reserved.append(lit)

print("Reserved words:", reserved)

# 3) Build a single negative‐lookahead for all reserved words
kw_pattern = "|".join(map(re.escape, reserved))
last_resort_pattern = rf"""^(?![-'"()])(?!\b(?:{kw_pattern})\b)[^()'"\s]+$"""
last_resort_pattern = rf"""^(?![-'"()])(?!(?:{kw_pattern})$)[^()'"\s]+$"""
last_resort_pattern_fixed = rf"""(?![-'"()])(?!(?:{kw_pattern})\b)(?!\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b)(?=[A-Za-z])[^()'"\s]+""".replace('/', r'\/')


print ("Keyword pattern:", kw_pattern)
print("Last resort pattern:", last_resort_pattern_fixed)

# 4) The “last‐resort” regex:
LAST_RESORT = re.compile(
    last_resort_pattern,
    re.VERBOSE
)

# 5) Test it
tests = ["foo", "bar123", "-baz", "(qux)", '"quux"', "name", "and", "xor", "print(aa)"]
for t in tests:
    print(f"{t!r}: ", bool(LAST_RESORT.match(t)))
