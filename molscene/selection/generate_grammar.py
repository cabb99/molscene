#!/usr/bin/env python3
"""
scripts/generate_grammar.py

Reads:
  - molscene/selection/grammar_template.lark
  - macros.json
  - keywords.json

Writes:
  - molscene/selection/grammar.lark

Usage:
  python scripts/generate_grammar.py
"""

import re
from pathlib import Path

import json   # pip install json5

TEMPLATE      = Path("molscene/selection/grammar_template.lark")
OUT_FILE      = Path("molscene/selection/grammar.lark")
MACROS_JSON   = Path("molscene/selection/macros.json")
KEYWORDS_JSON = Path("molscene/selection/keywords.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def make_token_block(tokens: dict) -> tuple[str,str]:
    """
    Returns (block_text, names_alternation)
    """
    lines = []
    names = []
    print(f"Processing tokens: {len(tokens)} categories")

    for category in tokens.keys():
        print(f"Processing category: {category}")
        for token in tokens[category]:
            print(f"Processing tokens: {token}")
            if token.startswith("_"):
                continue
            name = token.upper()
            token = tokens[category][token]
            print(token.keys())
            macro_token = f"{name.upper()}"
            macro_rule = f'{macro_token} : "{name}"'
            if "synonyms" in token and len(token["synonyms"]) > 0:
                macro_rule += " | " + " | ".join(f'"{syn}"' for syn in token["synonyms"])
            lines.append(macro_rule)
            names.append(macro_token)
    return "\n".join(lines), " | ".join(names)


def make_keywords_block(keywords: dict) -> tuple[str,str]:
    """
    Returns (block_text, names_alternation)
    """
    lines = []
    names = []
    # keywords is a dict of categories, each containing dicts of keywords
    for category in keywords.values():
        print(f"Processing category: {category['name']}")
        for kw in category.keys():
            tok = kw.upper()
            lines.append(f'{tok} : "{kw}"')
            names.append(tok)
    return "\n".join(lines), " | ".join(names)


def compute_last_token_pattern(grammar_text: str) -> str:
    """
    Scans all lines of the (already-injected) grammar for
    UPPERCASE_TOKEN : "lit" and collects every lit,
    then builds the negative-lookahead regex.
    """
    # Remove comments
    no_comments = re.sub(r'//.*', '', grammar_text)
    reserved = []
    for line in no_comments.splitlines():
        m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*:\s*(.+)$', line)
        if not m:
            continue
        for lit in re.findall(r'"([^"]+)"', m.group(2)):
            reserved.append(lit)
    # Build alternation pattern for reserved words
    kw_pat = "|".join(map(re.escape, reserved))
    # Compose the last-token regex pattern (no ^/$ anchors, use \b after reserved alternation)
    last_token_pattern = (
        r"(?![-'\"()])"  # not starting with these punctuations
        rf"(?!(?:{kw_pat})\\b)"  # not a reserved word
        r"(?!\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?\b)"  # not a number
        r"(?=[A-Za-z])"  # must start with a letter
        r"[^()'\"\\s]+"  # match token
    )
    return last_token_pattern


def main():
    # load template + JSON
    tpl_text  = TEMPLATE.read_text()
    macros    = load_json(MACROS_JSON)["macros"]
    keywords  = load_json(KEYWORDS_JSON)["keywords"]

    # build macro & keyword sections
    macros_block, macros_names = make_token_block(macros)
    kw_block,    kw_names      = make_token_block(keywords)

    # first round of replacements
    interim = (
        tpl_text
        .replace("<<MACROS>>", macros_block)
        .replace("<<MACROS_NAMES>>", macros_names)
        .replace("<<KEYWORDS>>", kw_block)
        .replace("<<KEYWORDS_NAMES>>", kw_names)
    )

    # print(macros_block)
    # compute last‐token pattern
    last_tok = compute_last_token_pattern(interim)

    # final injection
    final = interim.replace("<<LAST_TOKEN>>", last_tok)

    # write out
    OUT_FILE.write_text(final)
    print(f" Generated grammar at {OUT_FILE}")


if __name__ == "__main__":
    main()
