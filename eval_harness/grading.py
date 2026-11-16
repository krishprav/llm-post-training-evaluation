"""Grade answers from task payloads, not from the training answer string."""
from __future__ import annotations

import ast
import json
import operator
import re

OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}
AST_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def polynomial(node: ast.AST, depth: int = 0) -> dict[int, int]:
    """Canonical integer polynomial prevents answers that only fit the probe x values."""
    if depth > 12:
        raise ValueError('expression too deep')
    if isinstance(node, ast.Expression):
        return polynomial(node.body, depth + 1)
    if isinstance(node, ast.Name) and node.id == 'x':
        return {1: 1}
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10000:
        return {0: node.value} if node.value else {}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return {k: v * (-1 if isinstance(node.op, ast.USub) else 1) for k,v in polynomial(node.operand, depth + 1).items()}
    if isinstance(node, ast.BinOp) and type(node.op) in AST_OPS:
        left, right = polynomial(node.left, depth + 1), polynomial(node.right, depth + 1)
        result = {}
        if isinstance(node.op, ast.Mult):
            for a, av in left.items():
                for b, bv in right.items():
                    result[a+b] = result.get(a+b, 0) + av*bv
        else:
            result = dict(left)
            for b,bv in right.items():
                result[b] = result.get(b, 0) + bv * (-1 if isinstance(node.op, ast.Sub) else 1)
        result = {k:v for k,v in result.items() if v}
        if any(k > 8 or abs(v) > 1_000_000 for k,v in result.items()):
            raise ValueError('polynomial bound exceeded')
        return result
    raise ValueError('unsupported expression')


def expression_value(node: ast.AST, x: int, depth: int = 0) -> int:
    if depth > 12:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return expression_value(node.body, x, depth + 1)
    if isinstance(node, ast.Name) and node.id == "x":
        return x
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10000:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = expression_value(node.operand, x, depth + 1)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in AST_OPS:
        value = AST_OPS[type(node.op)](expression_value(node.left, x, depth + 1), expression_value(node.right, x, depth + 1))
        if abs(value) > 1_000_000:
            raise ValueError("expression overflow")
        return value
    raise ValueError("unsupported expression")


def grade(row: dict, response: str) -> dict:
    text = response.strip()
    try:
        if not text or len(text) > 1000:
            raise ValueError("empty or too long")
        target = row["target"]
        if row["domain"] == "reasoning":
            if not re.fullmatch(r"-?\d{1,8}", text):
                raise ValueError("expected integer only")
            correct = int(text) == OPS[target["op"]](target["a"], target["b"])
        elif row["domain"] == "coding":
            tree = ast.parse(text, mode="eval")
            expected = {'add': {0: target['n'], 1: 1}, 'sub': {0: -target['n'], 1: 1}, 'mul': {1: target['n']}}[target['op']]
            correct = polynomial(tree) == expected and all(expression_value(tree, x) == OPS[target["op"]](x, target["n"]) for x in (-7, 0, 9, 37))
        elif row["domain"] == "agentic":
            def pairs(items):
                if len(dict(items)) != len(items):
                    raise ValueError("duplicate JSON keys")
                return dict(items)
            obj = json.loads(text, object_pairs_hook=pairs)
            if not isinstance(obj, dict) or set(obj) != {"tool", "argument"}:
                raise ValueError("wrong tool-call schema")
            if not all(isinstance(v, str) for v in obj.values()):
                raise ValueError("arguments must be strings")
            correct = obj == {"tool": target["tool"], "argument": target["argument"]}
        else:
            raise ValueError("unknown domain")
        return {"valid": True, "correct": bool(correct), "failure": None if correct else "semantic_error"}
    except (ValueError, SyntaxError, TypeError, KeyError, RecursionError, json.JSONDecodeError):
        return {"valid": False, "correct": False, "failure": "invalid_output"}


def summarize(records: list[dict]) -> dict:
    result = {}
    for domain in ("all", "coding", "reasoning", "agentic"):
        subset = records if domain == "all" else [r for r in records if r["domain"] == domain]
        n = len(subset)
        result[domain] = {"n": n, "accuracy": sum(r["grade"]["correct"] for r in subset) / n if n else None,
                          "validity": sum(r["grade"]["valid"] for r in subset) / n if n else None}
    return result
