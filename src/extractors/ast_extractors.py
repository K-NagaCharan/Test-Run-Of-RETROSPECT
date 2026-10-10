"""
Language-specific AST Extractors and Feature Extractors for:
- Python (via built-in ast)
- Java (via javalang)
- JavaScript (via esprima)
- TypeScript (via node typescript script)
"""
import ast
import re
import json
import subprocess
from typing import Dict, Any, List, Optional
import javalang
import esprima

TS_PARSER_SCRIPT = """
const parser = require('@babel/parser');
const fs = require('fs');

const code = fs.readFileSync(0, 'utf-8');
let functionInfo = {
    name: '',
    parameters: [],
    node_types: [],
    statement_count: 0
};

try {
    const ast = parser.parse(code, {
        sourceType: 'unambiguous',
        plugins: ['typescript', 'jsx', 'classProperties', 'decorators-legacy']
    });

    function walk(node) {
        if (!node || typeof node !== 'object') return;
        if (node.type) {
            functionInfo.node_types.push(node.type);
            if (node.type === 'FunctionDeclaration' || node.type === 'MethodDefinition' || node.type === 'ArrowFunctionExpression' || node.type === 'FunctionExpression') {
                if (node.id && node.id.name && !functionInfo.name) {
                    functionInfo.name = node.id.name;
                } else if (node.key && node.key.name && !functionInfo.name) {
                    functionInfo.name = node.key.name;
                }
                if (node.params && functionInfo.parameters.length === 0) {
                    functionInfo.parameters = node.params.map(p => p.name || (p.left && p.left.name) || 'arg');
                }
            }
            if (node.type.endsWith('Statement')) {
                functionInfo.statement_count += 1;
            }
        }
        for (let key in node) {
            if (key !== 'loc' && key !== 'range' && key !== 'comments') {
                let child = node[key];
                if (Array.isArray(child)) {
                    child.forEach(walk);
                } else if (child && typeof child === 'object') {
                    walk(child);
                }
            }
        }
    }

    walk(ast);
} catch (e) {
    // on parse failure, functionInfo defaults
}
console.log(JSON.stringify(functionInfo));
"""

class ASTFeatures:
    def __init__(
        self,
        name: str = "",
        parameters: Optional[List[str]] = None,
        return_type: str = "",
        node_types: Optional[List[str]] = None,
        statement_count: int = 0,
        raw_source: str = ""
    ):
        self.name = name
        self.parameters = parameters or []
        self.return_type = return_type
        self.node_types = node_types or []
        self.statement_count = statement_count
        self.raw_source = raw_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "node_types": self.node_types[:50],  # truncate if very long
            "statement_count": self.statement_count
        }

def parse_python_ast(source_code: str) -> ASTFeatures:
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [arg.arg for arg in node.args.args]
                node_types = [type(n).__name__ for n in ast.walk(node)]
                stmt_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.stmt))
                ret_ann = ""
                if node.returns:
                    ret_ann = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(type(node.returns))
                return ASTFeatures(
                    name=node.name,
                    parameters=params,
                    return_type=ret_ann,
                    node_types=node_types,
                    statement_count=stmt_count,
                    raw_source=source_code
                )
    except Exception:
        pass
    # Fallback heuristic
    return fallback_parse(source_code, "python")

def parse_java_ast(source_code: str) -> ASTFeatures:
    try:
        # If method snippet only, wrap in dummy class
        wrapped = source_code
        if not re.search(r'\bclass\s+\w+', source_code):
            wrapped = f"public class DummyClass {{\n{source_code}\n}}"
        tokens = javalang.tokenizer.tokenize(wrapped)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            params = [param.name for param in node.parameters] if node.parameters else []
            ret_type = node.return_type.name if node.return_type else "void"
            node_types = [type(n).__name__ for _, n in node]
            stmt_count = len(node.body) if node.body else 0
            return ASTFeatures(
                name=node.name,
                parameters=params,
                return_type=ret_type,
                node_types=node_types,
                statement_count=stmt_count,
                raw_source=source_code
            )
    except Exception:
        pass
    return fallback_parse(source_code, "java")

def parse_javascript_ast(source_code: str) -> ASTFeatures:
    try:
        # Wrap expression if needed
        parsed = esprima.parseScript(source_code, tolerant=True)
        node_types = []
        name = ""
        params = []
        stmt_count = 0

        for node in parsed.body:
            if node.type == 'FunctionDeclaration':
                name = node.id.name if node.id else ""
                params = [p.name for p in node.params if hasattr(p, 'name')]
                stmt_count = len(node.body.body) if hasattr(node.body, 'body') else 0
                break
            elif node.type == 'VariableDeclaration':
                for decl in node.declarations:
                    if decl.init and decl.init.type in ('FunctionExpression', 'ArrowFunctionExpression'):
                        name = decl.id.name if hasattr(decl.id, 'name') else ""
                        params = [p.name for p in decl.init.params if hasattr(p, 'name')]
                        break
        # Walk node types
        def walk(n):
            if hasattr(n, 'type'):
                node_types.append(n.type)
                for k, v in n.items():
                    if isinstance(v, list):
                        for item in v:
                            walk(item)
                    elif hasattr(v, 'type'):
                        walk(v)
        walk(parsed)
        return ASTFeatures(
            name=name,
            parameters=params,
            return_type="",
            node_types=node_types,
            statement_count=stmt_count,
            raw_source=source_code
        )
    except Exception:
        pass
    return fallback_parse(source_code, "javascript")

def parse_typescript_ast(source_code: str) -> ASTFeatures:
    try:
        res = subprocess.run(
            ['node', '-e', TS_PARSER_SCRIPT],
            input=source_code,
            text=True,
            capture_output=True,
            timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            d = json.loads(res.stdout.strip())
            return ASTFeatures(
                name=d.get("name", ""),
                parameters=d.get("parameters", []),
                return_type="",
                node_types=d.get("node_types", []),
                statement_count=d.get("statement_count", 0),
                raw_source=source_code
            )
    except Exception:
        pass
    return fallback_parse(source_code, "typescript")

def fallback_parse(source_code: str, lang: str) -> ASTFeatures:
    # Regex fallback
    name = ""
    params = []
    m = re.search(r'(?:def|function|public|private|protected)?\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)', source_code)
    if m:
        name = m.group(1)
        raw_params = m.group(2).split(',')
        params = [p.strip().split()[-1] for p in raw_params if p.strip()]
    lines = [line.strip() for line in source_code.splitlines() if line.strip()]
    return ASTFeatures(
        name=name,
        parameters=params,
        return_type="",
        node_types=["FallbackToken"] * min(len(lines), 20),
        statement_count=len(lines),
        raw_source=source_code
    )

def extract_features(source_code: str, language: str) -> ASTFeatures:
    lang = language.lower()
    if lang == "python":
        return parse_python_ast(source_code)
    elif lang == "java":
        return parse_java_ast(source_code)
    elif lang in ("javascript", "js"):
        return parse_javascript_ast(source_code)
    elif lang in ("typescript", "ts"):
        return parse_typescript_ast(source_code)
    else:
        return fallback_parse(source_code, lang)
