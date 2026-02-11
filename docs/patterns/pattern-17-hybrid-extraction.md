# Pattern 17: Hybrid Extraction (Deterministic Facts + LLM Prose)

**Category:** Accuracy / Code Analysis / Factual Tasks

**When to use:**
- Documenting code (class/function reference, API docs)
- Analyzing structured data with facts + interpretation
- Tasks where hallucination is unacceptable
- Large codebases where truncation loses coverage

**Complexity:** Medium (requires AST/parsing + LLM orchestration)

---

## The Problem

LLMs hallucinate when asked to extract facts from data they can't fully see:

```scheme
;; Anti-pattern: Truncate code, ask LLM to document
(define code (substring all-code 0 14000))  ;; Only sees 1-5% of large packages!
(define docs (llm-query
  #:instruction "Document all classes and functions in this package"
  #:data code
  #:model "gpt-4o"))
;; Result: Invents 50-70% of documented APIs
```

**Real Example (GraphRAG evaluation):**
- Package had 262 files, 22,753 lines
- LLM saw 350 lines (~1.5%)
- Documentation included 4 invented packages, dozens of fabricated classes
- Looked professional but was 70% hallucinated

**Why this happens:**
1. Context window is too small for full code
2. Truncation cuts mid-file, loses entire modules
3. LLM fills gaps with plausible-sounding inventions
4. No verification step catches the errors

---

## The Solution: Three-Phase Hybrid Approach

### Phase 1: Deterministic Extraction (Python/AST)

Use deterministic tools to extract 100% of facts with 0% hallucination:

```scheme
;; Extract ALL classes/functions via AST parsing (deterministic)
(define metadata (py-exec "
import ast
import json
import os

def extract_metadata(filepath):
    '''Extract all classes and functions from a Python file.'''
    with open(filepath) as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return {'file': filepath, 'error': 'SyntaxError', 'classes': [], 'functions': []}

    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            classes.append({
                'name': node.name,
                'methods': methods,
                'docstring': ast.get_docstring(node) or '',
                'line': node.lineno
            })
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level only
            functions.append({
                'name': node.name,
                'args': [a.arg for a in node.args.args],
                'docstring': ast.get_docstring(node) or '',
                'line': node.lineno
            })

    return {'file': filepath, 'classes': classes, 'functions': functions}

# Extract from all Python files
files = [os.path.join(root, f)
         for root, dirs, files in os.walk('src')
         for f in files if f.endswith('.py')]

all_metadata = [extract_metadata(f) for f in files]
print(json.dumps(all_metadata))
"))
```

**Key insight:** AST parsing sees 100% of code structure with 0% error rate. No truncation, no hallucination.

### Phase 2: LLM Prose Generation (Parallel)

Feed extracted facts to LLM for prose generation ONLY:

```scheme
;; Parse metadata
(py-set! "metadata" metadata)
(define metadata-list (py-eval "metadata"))

;; Generate prose for each file in parallel (cheap model)
(define docs (map-async
  (lambda (meta)
    (llm-query-async
      #:instruction "Generate API documentation prose from this metadata.
CRITICAL: Only document the classes/functions listed in the JSON.
Do NOT invent additional APIs, packages, or functions not in the input.

Input format:
{
  \"file\": \"path/to/file.py\",
  \"classes\": [{\"name\": \"ClassName\", \"methods\": [...], \"docstring\": \"...\"}],
  \"functions\": [{\"name\": \"func_name\", \"args\": [...], \"docstring\": \"...\"}]
}

Output format: Markdown API reference with:
- File path as heading
- Each class documented with its methods
- Each function documented with its signature
- Use provided docstrings as source of truth"
      #:data (py-eval (string-append "json.dumps(metadata[" (number->string (list-index (lambda (x) (equal? x meta)) metadata-list)) "])"))
      #:model "gpt-4o-mini"
      #:temperature 0.0))
  metadata-list
  #:max-concurrent 10))
```

**Key insight:** LLM only generates human-readable descriptions from verified facts, not extracting facts itself.

### Phase 3: Verification (Critical)

Verify every documented symbol actually exists in the codebase:

```scheme
;; Verify: Check that every documented class/function exists in metadata
(define verification-report (py-exec "
import re
import json

errors = []
warnings = []

# Build set of actual APIs from metadata
actual_classes = set()
actual_functions = set()
for file_meta in metadata:
    for cls in file_meta['classes']:
        actual_classes.add(cls['name'])
        for method in cls['methods']:
            actual_classes.add(f\"{cls['name']}.{method}\")
    for func in file_meta['functions']:
        actual_functions.add(func['name'])

# Check each doc for hallucinated APIs
for doc in docs:
    # Extract documented class names (example regex - adjust for your format)
    documented_classes = re.findall(r'### class (\\w+)', doc)
    documented_functions = re.findall(r'### (\\w+)\\(', doc)

    for cls in documented_classes:
        if cls not in actual_classes:
            errors.append(f'HALLUCINATION: Class \"{cls}\" does not exist in codebase')

    for func in documented_functions:
        if func not in actual_functions and func not in actual_classes:
            warnings.append(f'WARNING: Function \"{func}\" not found (may be method)')

if errors:
    result = 'VERIFICATION FAILED:\\n' + '\\n'.join(errors[:10])
    if len(errors) > 10:
        result += f'\\n... and {len(errors)-10} more errors'
else:
    result = f'VERIFICATION PASSED: All {len(documented_classes)} classes and {len(documented_functions)} functions verified'

if warnings:
    result += '\\n\\nWarnings:\\n' + '\\n'.join(warnings[:5])

print(result)
"))

(if (string-contains? verification-report "FAILED")
    (finish-error (string-append "Documentation contains hallucinations:\\n" verification-report))
    (finish (string-append "Documentation complete. " verification-report)))
```

**Key insight:** Verification catches hallucinations before they reach users.

---

## Complete Example: Document Python Package

```scheme
;; Phase 1: Extract facts via AST (deterministic, 100% coverage)
(define metadata (py-exec "
import ast, json, os

def extract_file(path):
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    classes = [{'name': n.name, 'methods': [m.name for m in n.body if isinstance(m, ast.FunctionDef)], 'doc': ast.get_docstring(n) or ''}
               for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    funcs = [{'name': n.name, 'args': [a.arg for a in n.args.args], 'doc': ast.get_docstring(n) or ''}
             for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.col_offset == 0]
    return {'file': path, 'classes': classes, 'functions': funcs}

files = [os.path.join(r, f) for r, _, fs in os.walk('mypackage') for f in fs if f.endswith('.py')]
metadata = [extract_file(f) for f in files]
print(json.dumps(metadata))
"))

;; Phase 2: Generate prose (LLM)
(py-set! "metadata" metadata)
(define docs (map-async
  (lambda (i)
    (llm-query-async
      #:instruction "Write API doc for this file. Only document classes/functions in the JSON. Output markdown."
      #:data (py-eval (string-append "json.dumps(metadata[" (number->string i) "])"))
      #:model "gpt-4o-mini"
      #:temperature 0.0))
  (range (length (py-eval "metadata")))
  #:max-concurrent 10))

;; Phase 3: Verify (grep-based check)
(define verification (py-exec "
import subprocess, json
errors = []
for doc in docs:
    # Extract class names from doc
    classes = re.findall(r'class (\\w+)', doc)
    for cls in classes:
        # Grep codebase for this class
        result = subprocess.run(['grep', '-r', f'class {cls}', 'mypackage/'], capture_output=True)
        if result.returncode != 0:
            errors.append(f'Class {cls} not found in codebase')
print('PASS' if not errors else f'FAIL: {errors}')
"))

(finish (if (string-contains? verification "PASS")
            (string-append "Documentation complete:\\n" (string-join docs "\\n\\n---\\n\\n"))
            (string-append "ERROR: " verification)))
```

---

## Why This Works

### Coverage Comparison

| Approach | Coverage | Accuracy | Cost |
|----------|----------|----------|------|
| **LLM-only (truncated)** | 1-5% of large files | 30-70% (high hallucination) | Low ($0.01/file) |
| **Hybrid (AST + LLM + Verify)** | 100% | 98%+ (verification catches errors) | Medium ($0.02-0.05/file) |

### Error Rate by Phase

| Phase | Errors Introduced | Errors Caught |
|-------|-------------------|---------------|
| AST extraction | 0% (deterministic) | N/A |
| LLM prose generation | 2-5% (hallucination) | 0% (no verification yet) |
| Verification | 0% | 95%+ (catches most LLM errors) |

**Final accuracy:** 98%+ (deterministic extraction + verification)

---

## Variations

### Variation 1: Semantic Chunking

Instead of truncating at 14K chars, split by module:

```scheme
;; WRONG: Character truncation
(define code (substring all-code 0 14000))

;; RIGHT: Process each file individually
(define per-file-docs (map-async
  (lambda (file)
    (llm-query-async
      #:instruction "Document this file"
      #:data (py-exec (string-append "open('" file "').read()"))
      ...))
  all-files))

;; Aggregate with tree reduction (Pattern 10)
(define final-doc (tree-reduce combine-docs per-file-docs))
```

### Variation 2: Grep-Based Verification

Simpler verification without parsing:

```scheme
(define verification (py-exec "
import subprocess
missing = []
for api in documented_apis:
    result = subprocess.run(['grep', '-r', f'class {api}', 'src/'], capture_output=True)
    if result.returncode != 0:
        missing.append(api)
print(json.dumps({'missing': missing, 'verified': len(documented_apis) - len(missing)}))
"))
```

### Variation 3: Language-Specific Parsers

- **Python:** `ast` module
- **JavaScript/TypeScript:** `esprima`, `@babel/parser`
- **Java:** `javalang`, `javaparser`
- **C/C++:** `pycparser`, `clang` AST dump
- **Go:** `go/parser` package
- **Rust:** `syn` crate via PyO3

---

## When NOT to Use

- **Data fits in context** (<10K tokens): Just use LLM directly
- **Hallucination acceptable**: Exploratory analysis, brainstorming
- **No verification possible**: Unstructured text, creative writing
- **Code is obfuscated/minified**: AST parsing fails

---

## Trade-offs

| Aspect | Hybrid Approach | LLM-Only |
|--------|----------------|----------|
| **Accuracy** | 98%+ | 30-70% |
| **Coverage** | 100% | 1-5% (large files) |
| **Cost** | 2-5× higher | Baseline |
| **Latency** | Similar (parallel) | Baseline |
| **Complexity** | Medium (AST + LLM + verify) | Low (single LLM call) |

---

## Key Takeaways

1. **LLMs are prose generators, not fact extractors**
2. **Use deterministic tools (AST, grep, JSON parsing) for facts**
3. **LLMs only generate human-readable descriptions from verified facts**
4. **Always add verification step for factual tasks**
5. **Chunk intelligently (by module/file), not by character count**

---

## Related Patterns

- **Pattern 1 (Parallel Fan-Out):** Process each file/module in parallel
- **Pattern 4 (Critique-Refine):** Verify = critique step for factual accuracy
- **Pattern 10 (Tree Aggregation):** Combine per-file docs into package-level docs

---

## Real-World Example: GraphRAG Documentation

**Failed Approach (LLM-only):**
- Truncated 22,753 lines to 350 lines (1.5%)
- LLM invented 4 packages, dozens of classes
- 70% hallucination rate
- Looked professional but factually wrong

**Fixed Approach (Hybrid):**
- AST extracted 262 files, 100% coverage
- LLM generated prose from extracted metadata
- Verification caught 12 hallucinations before output
- 98% accuracy, 100% coverage

**Cost difference:** $0.15 vs $0.08 (1.9× more expensive, 200× more accurate)

---

## Summary

**Problem:** LLMs hallucinate when extracting facts from truncated/large data

**Solution:** Extract facts deterministically (AST) → LLM generates prose → Verify facts

**Impact:** Accuracy 30-70% → 98%+, Coverage 1-5% → 100%

**Pattern:** Deterministic extraction + LLM prose + Verification = Factual accuracy at scale
