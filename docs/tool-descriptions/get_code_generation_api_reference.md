Get condensed API reference for code-generating sub-models.

When using Pattern 2 (Code Generation), sub-models don't automatically know
the rlm-scheme API. Call this tool and include its output in your unsafe-raw-query
#:data parameter so the sub-model generates correct syntax.

This returns a minimal reference (~200 lines) optimized for inclusion in prompts.
For the full guide with strategies and examples, use get_usage_guide instead.
