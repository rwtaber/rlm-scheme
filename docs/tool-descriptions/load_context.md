Load input data into the sandbox. Available as `context` in Scheme and Python.

Args:
    data: Text data to load (documents, code, CSV, JSON, etc.)
    name: Optional name for this context slot (e.g., "gwas-data", "expression").
          Use get-context to retrieve named contexts later.

Named context slots (improvement #5) allow managing multiple datasets:
- load_context(gwas_csv, "gwas-data")
- load_context(expr_csv, "expression-data")
- Later in Scheme: (get-context "gwas-data") or (get-context "expression-data")

Strategy considerations after loading:
- Data >100KB? → Use fan-out-aggregate (chunk via py-exec, parallel processing with map-async)
- Unknown structure? → Use meta-orchestration (model inspects sample, generates analysis code)
- Hierarchical? → Use recursive-spawn (delegate to specialized sandboxes)

Call plan_strategy(task) for custom strategy recommendations.
