Get condensed API reference for code-generating sub-models.

When using code generation strategies, sub-models don't automatically know
the rlm-scheme API. Call this tool and include its output in your LLM prompt
so the sub-model generates correct Scheme syntax.

This returns a minimal reference (~200 lines) optimized for inclusion in prompts.
Includes core primitives and combinator library.

For the full guide with examples, use get_usage_guide instead.
