# Model Registry: Configuration-Driven Model Selection

**Status:** Proposed
**Date:** 2026-02-12
**Priority:** High - Architectural Improvement

---

## Executive Summary

Replace hardcoded model defaults with a configuration-driven model registry that supports semantic model names, multi-provider support, and explicit model selection.

**Current Problem:**
- Hardcoded default model (`"gpt-4o"`) in code
- Model IDs like `"gpt-4o-mini"` are opaque and provider-specific
- No way to change models without code changes
- Hard to audit model usage
- Single provider (OpenAI) baked into architecture

**Proposed Solution:**
- Configuration file defines available models with semantic names
- Code uses semantic names: `"fast"`, `"balanced"`, `"quality"`
- Multi-provider support (OpenAI, Anthropic, future providers)
- Explicit model selection enforced
- Easy to audit and control costs

---

## Table of Contents

- [Problems with Current System](#problems-with-current-system)
- [Proposed Architecture](#proposed-architecture)
- [Configuration Schema](#configuration-schema)
- [Implementation Plan](#implementation-plan)
- [Migration Strategy](#migration-strategy)
- [Benefits](#benefits)
- [Examples](#examples)
- [Open Questions](#open-questions)

---

## Problems with Current System

### 1. Hardcoded Model Defaults

**Current code:**
```python
model = model or "gpt-4o"  # Hardcoded!
```

**Issues:**
- ❌ Can't change default model without code changes
- ❌ Different environments (dev/prod) must use same models
- ❌ No way to restrict available models
- ❌ Default is implicit and hidden

### 2. Opaque Model Identifiers

**Current usage:**
```scheme
(llm-query #:model "gpt-4o-mini" ...)  ; What does this mean?
```

**Issues:**
- ❌ "gpt-4o-mini" doesn't convey intent (fast? cheap? quality?)
- ❌ Changing models requires finding and replacing all instances
- ❌ Provider-specific names leak into orchestration code
- ❌ Unclear why a particular model was chosen

### 3. Single Provider Lock-In

**Current architecture:**
- Only OpenAI API supported
- Model names are OpenAI-specific
- Can't easily switch to Claude, Gemini, etc.

**Issues:**
- ❌ Vendor lock-in
- ❌ Can't mix providers (e.g., Claude for planning, OpenAI for bulk work)
- ❌ Can't take advantage of new providers

### 4. No Cost/Usage Control

**Issues:**
- ❌ Can't restrict which models are available
- ❌ No way to enforce "only cheap models in dev"
- ❌ Hard to audit which models are being used where
- ❌ No visibility into model usage patterns

### 5. Implicit Model Selection

**Current pattern:**
```scheme
;; What model does this use?
(llm-query #:instruction "..." #:data "...")
;; Answer: gpt-4o (hardcoded default)
```

**Issues:**
- ❌ Hidden default makes code hard to understand
- ❌ Can't easily see what models a strategy uses
- ❌ Changing default affects all code

---

## Proposed Architecture

### Core Concept: Semantic Model Names

Instead of:
```scheme
(llm-query #:model "gpt-4o-mini" ...)  ; Opaque provider-specific ID
```

Use:
```scheme
(llm-query #:model "fast" ...)  ; Semantic name shows intent
```

**Semantic tiers:**
- `"fast"` - Cheap, quick, simple tasks (extraction, classification)
- `"balanced"` - Good quality/cost tradeoff (general tasks)
- `"quality"` - Highest quality (synthesis, complex reasoning)
- `"reasoning"` - Specialized reasoning models (o1, o3)
- Custom tiers as needed

### Configuration File Defines Mappings

**models.json** (or section in .mcp.json):
```json
{
  "models": {
    "fast": {
      "provider": "openai",
      "model_id": "gpt-4o-mini",
      "cost_per_1m_input": 150,
      "cost_per_1m_output": 600,
      "description": "Fast and cheap, for simple extraction/classification"
    },
    "balanced": {
      "provider": "openai",
      "model_id": "gpt-4o",
      "cost_per_1m_input": 2500,
      "cost_per_1m_output": 10000,
      "description": "Best balance for most tasks"
    },
    "quality": {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-5",
      "cost_per_1m_input": 3000,
      "cost_per_1m_output": 15000,
      "description": "Highest quality, use for critical synthesis"
    },
    "reasoning": {
      "provider": "openai",
      "model_id": "o1-mini",
      "cost_per_1m_input": 3000,
      "cost_per_1m_output": 12000,
      "description": "Specialized reasoning for complex problems"
    }
  },
  "default_model": "balanced",
  "require_explicit": false,
  "allow_literal_ids": true
}
```

### Multi-Provider Support

**Provider abstraction layer:**
```python
class ModelProvider(ABC):
    @abstractmethod
    def call(self, instruction, data, **kwargs) -> dict:
        pass

class OpenAIProvider(ModelProvider):
    def call(self, instruction, data, **kwargs) -> dict:
        # OpenAI API call
        pass

class AnthropicProvider(ModelProvider):
    def call(self, instruction, data, **kwargs) -> dict:
        # Anthropic API call
        pass

# Registry maps semantic names to (provider, model_id)
model_registry = {
    "fast": (OpenAIProvider(), "gpt-4o-mini"),
    "quality": (AnthropicProvider(), "claude-sonnet-4-5"),
}
```

---

## Configuration Schema

### Option 1: Embedded in .mcp.json

```json
{
  "mcpServers": {
    "rlm-scheme": {
      "command": "...",
      "args": [...],
      "cwd": "...",
      "models": {
        "fast": {
          "provider": "openai",
          "model_id": "gpt-4o-mini",
          "cost_per_1m_input": 150,
          "cost_per_1m_output": 600,
          "description": "Fast and cheap"
        },
        "balanced": {
          "provider": "openai",
          "model_id": "gpt-4o",
          "cost_per_1m_input": 2500,
          "cost_per_1m_output": 10000,
          "description": "Best balance"
        },
        "quality": {
          "provider": "anthropic",
          "model_id": "claude-sonnet-4-5",
          "cost_per_1m_input": 3000,
          "cost_per_1m_output": 15000,
          "description": "Highest quality"
        }
      },
      "model_settings": {
        "default_model": "balanced",
        "require_explicit": false,
        "allow_literal_ids": true,
        "warning_on_literal": true
      }
    }
  }
}
```

### Option 2: Separate models.json

**Pros:**
- Easier to share/version model configs
- Can have multiple configs (dev, prod, cost-limited)
- Cleaner separation of concerns

**models.json:**
```json
{
  "version": "1.0",
  "models": {
    "fast": {
      "provider": "openai",
      "model_id": "gpt-4o-mini",
      "cost_per_1m_input": 150,
      "cost_per_1m_output": 600,
      "description": "Fast and cheap, for simple extraction/classification",
      "max_tokens_default": 4096,
      "supports_vision": true,
      "supports_json_mode": true
    },
    "balanced": {
      "provider": "openai",
      "model_id": "gpt-4o",
      "cost_per_1m_input": 2500,
      "cost_per_1m_output": 10000,
      "description": "Best balance for most tasks",
      "max_tokens_default": 16384,
      "supports_vision": true,
      "supports_json_mode": true
    },
    "quality": {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-5",
      "cost_per_1m_input": 3000,
      "cost_per_1m_output": 15000,
      "description": "Highest quality, use for critical synthesis",
      "max_tokens_default": 8192,
      "supports_vision": true,
      "supports_json_mode": false
    },
    "reasoning": {
      "provider": "openai",
      "model_id": "o1-mini",
      "cost_per_1m_input": 3000,
      "cost_per_1m_output": 12000,
      "description": "Specialized reasoning for complex problems",
      "max_tokens_default": 65536,
      "supports_vision": false,
      "supports_json_mode": false,
      "supports_temperature": false
    }
  },
  "settings": {
    "default_model": "balanced",
    "require_explicit": false,
    "allow_literal_ids": true,
    "warning_on_literal": true,
    "cost_tracking": true
  },
  "provider_configs": {
    "openai": {
      "api_key_env": "OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1"
    },
    "anthropic": {
      "api_key_env": "ANTHROPIC_API_KEY",
      "base_url": "https://api.anthropic.com"
    }
  }
}
```

### Configuration Fields

**Model Definition:**
- `provider`: "openai" | "anthropic" | "custom"
- `model_id`: Provider-specific model identifier
- `cost_per_1m_input`: Cost per 1M input tokens (in cents)
- `cost_per_1m_output`: Cost per 1M output tokens (in cents)
- `description`: Human-readable description
- `max_tokens_default`: Default max tokens for this model
- `supports_vision`: Boolean - supports image inputs
- `supports_json_mode`: Boolean - supports JSON mode
- `supports_temperature`: Boolean - supports temperature parameter

**Settings:**
- `default_model`: Semantic name to use if none specified
- `require_explicit`: If true, error if no model specified (forces intentionality)
- `allow_literal_ids`: If true, allow using literal model IDs (backward compat)
- `warning_on_literal`: If true, warn when literal IDs are used
- `cost_tracking`: Enable automatic cost tracking

**Provider Configs:**
- `api_key_env`: Environment variable for API key
- `base_url`: API base URL (for custom endpoints)

---

## Implementation Plan

### Phase 1: Configuration Loading (Week 1)

**Goals:**
- Load model configuration from file
- Validate configuration schema
- Maintain backward compatibility

**Tasks:**
1. Define JSON schema for model configuration
2. Create `ModelRegistry` class to load and manage config
3. Add configuration validation
4. Load config on server startup
5. Add fallback for missing config (use current hardcoded defaults)

**Files:**
- `mcp_server.py` - Add ModelRegistry class
- `models.json` - Default configuration
- `docs/model-configuration.md` - Documentation

**Code:**
```python
class ModelRegistry:
    def __init__(self, config_path: str = "models.json"):
        self.config = self._load_config(config_path)
        self._validate_config()
        self.models = self.config["models"]
        self.settings = self.config.get("settings", {})

    def resolve_model(self, model_name: str | None) -> tuple[str, str]:
        """Returns (provider, model_id) for given semantic name or literal ID."""
        if not model_name:
            # No model specified
            if self.settings.get("require_explicit"):
                raise ValueError("Model must be explicitly specified")
            model_name = self.settings.get("default_model", "balanced")

        # Check if semantic name
        if model_name in self.models:
            model_config = self.models[model_name]
            return (model_config["provider"], model_config["model_id"])

        # Check if literal ID allowed
        if self.settings.get("allow_literal_ids"):
            if self.settings.get("warning_on_literal"):
                print(f"[rlm] Warning: Using literal model ID '{model_name}'. "
                      f"Consider using semantic name instead.", file=sys.stderr)
            return ("openai", model_name)  # Assume OpenAI for backward compat

        raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

    def get_model_info(self, model_name: str) -> dict:
        """Get full configuration for a model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        return self.models[model_name]
```

### Phase 2: Multi-Provider Support (Week 2)

**Goals:**
- Abstract provider interface
- Implement OpenAI provider
- Implement Anthropic provider
- Route calls to correct provider

**Tasks:**
1. Create `ModelProvider` abstract base class
2. Implement `OpenAIProvider` (extract from existing code)
3. Implement `AnthropicProvider` (new)
4. Update `_call_llm` to route through registry
5. Handle provider-specific features (vision, JSON mode, temperature)

**Files:**
- `model_providers.py` - New file with provider abstractions
- `mcp_server.py` - Update to use providers

**Code:**
```python
# model_providers.py
from abc import ABC, abstractmethod

class ModelProvider(ABC):
    @abstractmethod
    def call(self, model_id: str, instruction: str, data: str,
             temperature: float | None = None,
             max_tokens: int | None = None,
             json_mode: bool = False,
             images: list[str] | None = None) -> dict:
        """
        Call the model and return response.

        Returns:
            {"text": str, "prompt_tokens": int, "completion_tokens": int}
        """
        pass

    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports feature (vision, json_mode, temperature)."""
        pass

class OpenAIProvider(ModelProvider):
    def call(self, model_id, instruction, data, **kwargs) -> dict:
        # Existing OpenAI implementation
        pass

    def supports_feature(self, feature: str) -> bool:
        return feature in ["vision", "json_mode", "temperature"]

class AnthropicProvider(ModelProvider):
    def call(self, model_id, instruction, data, **kwargs) -> dict:
        import anthropic
        client = anthropic.Anthropic()

        # Convert to Anthropic message format
        messages = []
        if instruction:
            # Anthropic uses system parameter, not system message
            system = instruction
        messages.append({"role": "user", "content": data or "(no data)"})

        # Handle images (Anthropic format different from OpenAI)
        if kwargs.get("images"):
            # TODO: Convert image format
            pass

        response = client.messages.create(
            model=model_id,
            system=system if instruction else None,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 1.0) if kwargs.get("temperature") is not None else 1.0,
        )

        return {
            "text": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    def supports_feature(self, feature: str) -> bool:
        return feature in ["vision", "temperature"]
```

### Phase 3: Update Planner to Use Semantic Names (Week 2)

**Goals:**
- Update planner to use semantic model names
- Update documentation and examples
- Maintain backward compatibility

**Tasks:**
1. Update `plan_strategy` to use semantic names
2. Update planner prompts with semantic model names
3. Update all examples in README
4. Update tests

**Changes:**
```python
# OLD
if priority == "cost":
    planner_model = "gpt-4o-mini"
elif priority == "quality":
    planner_model = "gpt-4o"
else:
    planner_model = "gpt-4o"

# NEW
if priority == "cost":
    planner_model = "fast"
elif priority == "quality":
    planner_model = "quality"
else:
    planner_model = "balanced"
```

**Update planner-prompt.md:**
```markdown
## Model Selection (Semantic Tiers)

| Tier | Cost | Characteristics | Use When |
|------|------|-----------------|----------|
| fast | ~$0.0002/1K | Quick, simple tasks | Extraction, classification, bulk work |
| balanced | ~$0.005/1K | Good quality/cost | General orchestration, most tasks |
| quality | ~$0.01/1K | Highest quality | Critical synthesis, complex reasoning |
| reasoning | ~$0.008/1K | Specialized reasoning | Planning, strategic decisions |

**Use semantic names in your strategies:**
```scheme
;; Good - intent is clear
(llm-query #:model "fast" ...)

;; Avoid - opaque provider-specific ID
(llm-query #:model "gpt-4o-mini" ...)
```
```

### Phase 4: Cost Tracking and Auditing (Week 3)

**Goals:**
- Track model usage by semantic name
- Report costs per model tier
- Enable cost auditing

**Tasks:**
1. Add cost tracking to ModelRegistry
2. Report usage in `get_status()`
3. Add cost breakdown in execution summary
4. Create cost analysis tools

**Code:**
```python
class ModelRegistry:
    def __init__(self, config_path: str = "models.json"):
        # ... existing init ...
        self.usage_stats = {
            model_name: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            for model_name in self.models.keys()
        }

    def track_usage(self, model_name: str, input_tokens: int, output_tokens: int):
        """Track usage for cost analysis."""
        if model_name not in self.usage_stats:
            return

        stats = self.usage_stats[model_name]
        stats["calls"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens

        # Calculate cost
        model_config = self.models[model_name]
        cost_input = (input_tokens / 1_000_000) * model_config["cost_per_1m_input"] / 100
        cost_output = (output_tokens / 1_000_000) * model_config["cost_per_1m_output"] / 100
        stats["cost"] += cost_input + cost_output

    def get_usage_report(self) -> dict:
        """Get usage breakdown by model tier."""
        return {
            "total_calls": sum(s["calls"] for s in self.usage_stats.values()),
            "total_cost": sum(s["cost"] for s in self.usage_stats.values()),
            "by_model": self.usage_stats
        }
```

### Phase 5: Documentation and Migration (Week 3)

**Goals:**
- Complete documentation
- Create migration guide
- Update all examples
- Add warnings for deprecated patterns

**Tasks:**
1. Write comprehensive documentation
2. Create migration guide from literal IDs to semantic names
3. Update all README examples
4. Add deprecation warnings
5. Create video/tutorial on new system

**Documentation files:**
- `docs/model-configuration.md` - Configuration reference
- `docs/model-migration-guide.md` - Migration instructions
- `docs/multi-provider-setup.md` - Setting up Anthropic/other providers

---

## Migration Strategy

### Backward Compatibility Approach

**Phase 1: Additive (No Breaking Changes)**
- Model registry optional (fallback to hardcoded defaults)
- Literal model IDs still work
- Semantic names work alongside literal IDs
- Add warnings but don't break existing code

**Phase 2: Deprecation Warnings**
- Warn when using literal model IDs
- Encourage migration to semantic names
- Update documentation to show new pattern
- Provide migration tool: `python migrate_models.py`

**Phase 3: Strict Mode (Optional)**
- `require_explicit: true` - Error if no model specified
- `allow_literal_ids: false` - Error on literal IDs
- Users opt-in to strict mode
- Helps enforce best practices

### Migration Tool

**migrate_models.py:**
```python
#!/usr/bin/env python3
"""Migrate Scheme code from literal model IDs to semantic names."""

import re
import sys

MODEL_MAPPING = {
    "gpt-4o-mini": "fast",
    "gpt-4.1-nano": "fast",
    "gpt-4o": "balanced",
    "gpt-4": "balanced",
    "claude-sonnet-4-5": "quality",
    "o1-mini": "reasoning",
}

def migrate_file(filepath: str):
    with open(filepath, 'r') as f:
        content = f.read()

    for old_model, new_model in MODEL_MAPPING.items():
        # Replace #:model "literal-id" with #:model "semantic-name"
        content = re.sub(
            rf'#:model\s+"{old_model}"',
            f'#:model "{new_model}"',
            content
        )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Migrated: {filepath}")

if __name__ == "__main__":
    for filepath in sys.argv[1:]:
        migrate_file(filepath)
```

### Example Configs for Different Environments

**dev.models.json** (Cost-limited for development):
```json
{
  "models": {
    "fast": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "balanced": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "quality": {"provider": "openai", "model_id": "gpt-4o"}
  },
  "settings": {
    "default_model": "fast",
    "require_explicit": true
  }
}
```

**prod.models.json** (Quality-optimized for production):
```json
{
  "models": {
    "fast": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "balanced": {"provider": "openai", "model_id": "gpt-4o"},
    "quality": {"provider": "anthropic", "model_id": "claude-sonnet-4-5"},
    "reasoning": {"provider": "openai", "model_id": "o1-mini"}
  },
  "settings": {
    "default_model": "balanced",
    "require_explicit": false
  }
}
```

---

## Benefits

### 1. Explicit and Auditable ✅

**Before:**
```scheme
(llm-query #:instruction "..." #:data "...")  ; What model? Who knows!
```

**After:**
```scheme
(llm-query #:model "fast" #:instruction "..." #:data "...")  ; Clear!
```

### 2. Semantic Clarity ✅

**Before:**
```scheme
(llm-query #:model "gpt-4o-mini" ...)  ; Why this model?
```

**After:**
```scheme
(llm-query #:model "fast" ...)  ; Intent is clear: bulk work, cost-optimized
```

### 3. Multi-Provider Support ✅

**Before:**
- Only OpenAI
- Vendor lock-in

**After:**
```json
"quality": {
  "provider": "anthropic",  // Use Claude for quality
  "model_id": "claude-sonnet-4-5"
}
```

### 4. Environment-Specific Configuration ✅

**Development:**
```bash
ln -s dev.models.json models.json  # Use cheap models
```

**Production:**
```bash
ln -s prod.models.json models.json  # Use quality models
```

### 5. Cost Control ✅

**Restrict to cheap models:**
```json
{
  "models": {
    "fast": {"model_id": "gpt-4o-mini"},
    "balanced": {"model_id": "gpt-4o-mini"}
  }
}
```

### 6. Easy Model Swapping ✅

**Change provider without code changes:**
```json
// Switch from OpenAI to Anthropic
"balanced": {
  "provider": "anthropic",
  "model_id": "claude-sonnet-4-5"
}
```

### 7. Cost Tracking and Analysis ✅

**Automatic cost breakdown:**
```
Model Usage Report:
  fast:     1000 calls, $0.15 (80% of calls, 10% of cost)
  balanced: 200 calls, $1.20 (16% of calls, 80% of cost)
  quality:  50 calls, $0.15 (4% of calls, 10% of cost)
Total: 1250 calls, $1.50
```

---

## Examples

### Example 1: Simple Configuration

**models.json:**
```json
{
  "models": {
    "fast": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "balanced": {"provider": "openai", "model_id": "gpt-4o"},
    "quality": {"provider": "openai", "model_id": "gpt-4o"}
  },
  "settings": {
    "default_model": "balanced"
  }
}
```

**Usage:**
```scheme
;; Use semantic names
(define summaries
  (map-async
    (lambda (doc)
      (llm-query #:model "fast" #:instruction "Summarize" #:data doc))
    documents))

;; Synthesis with quality model
(define final-report
  (llm-query
    #:model "quality"
    #:instruction "Synthesize comprehensive report"
    #:data (string-join summaries)))
```

### Example 2: Multi-Provider Setup

**models.json:**
```json
{
  "models": {
    "fast": {
      "provider": "openai",
      "model_id": "gpt-4o-mini",
      "description": "Fast OpenAI model for bulk work"
    },
    "balanced": {
      "provider": "openai",
      "model_id": "gpt-4o",
      "description": "Balanced OpenAI model"
    },
    "quality": {
      "provider": "anthropic",
      "model_id": "claude-sonnet-4-5",
      "description": "High-quality Claude model for synthesis"
    },
    "reasoning": {
      "provider": "openai",
      "model_id": "o1-mini",
      "description": "Reasoning model for complex planning"
    }
  },
  "settings": {
    "default_model": "balanced",
    "require_explicit": true
  },
  "provider_configs": {
    "openai": {
      "api_key_env": "OPENAI_API_KEY"
    },
    "anthropic": {
      "api_key_env": "ANTHROPIC_API_KEY"
    }
  }
}
```

**Usage:**
```scheme
;; OpenAI for extraction
(define extractions
  (map-async
    (lambda (item) (llm-query #:model "fast" ...))
    items))

;; Claude for high-quality synthesis
(define synthesis
  (llm-query #:model "quality" ...))

;; o1 for strategic planning
(define plan
  (llm-query #:model "reasoning" ...))
```

### Example 3: Cost-Limited Development Environment

**dev.models.json:**
```json
{
  "models": {
    "fast": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "balanced": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "quality": {"provider": "openai", "model_id": "gpt-4o-mini"}
  },
  "settings": {
    "default_model": "fast",
    "require_explicit": true,
    "cost_warning_threshold": 1.00
  }
}
```

**Result:**
- All semantic tiers map to cheapest model
- Forces explicit model selection
- Warns if cost exceeds $1.00
- Code doesn't change between dev and prod!

### Example 4: Updated Planner

**planner using semantic names:**
```python
def plan_strategy(...):
    if priority == "cost":
        planner_model = "fast"
        max_tokens = 10000
    elif priority == "quality":
        planner_model = "reasoning"  # Use o1-mini for complex planning
        max_tokens = 32000
    else:
        planner_model = "balanced"
        max_tokens = 15000
```

**User can change what "quality" means:**
```json
// Use Claude for quality planning
"quality": {
  "provider": "anthropic",
  "model_id": "claude-opus-4-6"
}
```

---

## Open Questions

### 1. Configuration Location

**Option A: Single models.json in project root**
- Pro: Simple, one place to look
- Con: Mixes with code

**Option B: ~/.rlm-scheme/models.json (user-level)**
- Pro: User preference, global across projects
- Con: Less visible, harder to version

**Option C: Both (project overrides user)**
- Pro: Best of both worlds
- Con: More complexity

**Recommendation:** Start with Option A (project-level), add Option C later.

### 2. Default Model Behavior

**Option A: Require explicit model selection**
- Pro: Forces intentionality
- Con: More verbose

**Option B: Have sensible default**
- Pro: Less verbose
- Con: Hidden decisions

**Recommendation:** Option B with config flag to enable Option A (strict mode).

### 3. Literal Model IDs

**Should we allow literal model IDs alongside semantic names?**

**Option A: Allow both**
- Pro: Flexibility, backward compatible
- Con: Mixed styles, harder to audit

**Option B: Semantic names only**
- Pro: Consistent, auditable
- Con: Breaking change, less flexible for experimentation

**Recommendation:** Option A with warnings, deprecate over time.

### 4. Provider Configuration

**Where should API keys be configured?**

**Option A: Environment variables (current)**
- Pro: Secure, standard practice
- Con: Not in config file

**Option B: Config file with env var references**
```json
"provider_configs": {
  "openai": {
    "api_key_env": "OPENAI_API_KEY"  // Reference to env var
  }
}
```
- Pro: Explicit in config
- Con: Still need env vars

**Recommendation:** Option B (best of both worlds).

### 5. Model Aliases

**Should we support custom aliases?**
```json
{
  "models": {
    "fast": {...},
    "balanced": {...},
    "quality": {...}
  },
  "aliases": {
    "cheap": "fast",
    "expensive": "quality",
    "default": "balanced"
  }
}
```

**Recommendation:** Not for MVP, but good future enhancement.

---

## Success Metrics

### Quantitative
- ✅ **Configuration coverage:** % of calls using semantic names (target: >90%)
- ✅ **Model diversity:** % using non-default models (target: >30%)
- ✅ **Cost tracking:** Accurate cost attribution per model tier
- ✅ **Zero downtime:** Backward compatibility maintained

### Qualitative
- ✅ User reports: "Model selection is now explicit and clear"
- ✅ Easier to audit: "I can see which models are used where"
- ✅ Multi-provider success: "I'm using Claude for quality tasks"
- ✅ Cost control: "I restricted dev environment to cheap models"

---

## Implementation Timeline

**Week 1: Configuration Loading**
- Define schema
- Implement ModelRegistry
- Load and validate config
- Maintain backward compatibility

**Week 2: Multi-Provider Support**
- Abstract provider interface
- Implement OpenAI provider
- Implement Anthropic provider
- Update planner to use semantic names

**Week 3: Polish and Documentation**
- Cost tracking and auditing
- Complete documentation
- Migration guide
- Examples and tutorials

**Total:** 3 weeks for full implementation

---

## Next Steps

1. **Review and approval** - Get feedback on this plan
2. **Create models.json schema** - Define the configuration format
3. **Implement ModelRegistry** - Core configuration management
4. **Add provider abstraction** - Multi-provider support
5. **Update planner** - Use semantic names
6. **Documentation** - Complete migration guide
7. **Testing** - Ensure backward compatibility

---

## Conclusion

The model registry provides a **configuration-driven, explicit, and auditable** system for model selection. It addresses all the problems with hardcoded defaults while maintaining backward compatibility.

**Key Benefits:**
- ✅ Semantic model names (intent-revealing)
- ✅ Multi-provider support (no vendor lock-in)
- ✅ Environment-specific configuration (dev vs prod)
- ✅ Cost control and auditing
- ✅ Easy model swapping (no code changes)
- ✅ Backward compatible (smooth migration)

**This is a significant architectural improvement that will make the system more flexible, auditable, and maintainable.**
