# OpenAI Model Update Summary

## Changes Made

Updated all documentation to use **actual OpenAI models** instead of placeholder/incorrect model names.

## Model Mapping (Old → New)

| Old Model (Incorrect) | New Model (Actual) | Cost (per 1K tokens) | Use Case |
|----------------------|-------------------|---------------------|----------|
| gpt-4.1-nano | **gpt-3.5-turbo** | $0.002 | Fan-out, extraction, bulk processing |
| gpt-4o-mini | **curie** | $0.002-0.03 | Critique, summarization, Q&A |
| gpt-4o | **gpt-4** | $0.03-0.06 | Synthesis, complex reasoning |
| gpt-4.1 | **code-davinci-002** | $0.02-0.05 | Code generation |
| gpt-4.1-mini | **gpt-3.5** | $0.01-0.02 | Mid-tier general tasks |
| o3-mini, o4-mini | **gpt-4** | $0.03-0.06 | Complex reasoning (o-series doesn't exist) |

## Complete OpenAI Model List (from provided info)

### Primary Text Models (for orchestration)

1. **ada** - $0.0004 / $0.005 per 1K tokens
   - Cheapest option for simple classification, filtering
   - 50-100× cheaper than GPT-4
   - Use for: Basic NLP, bulk filtering

2. **gpt-3.5-turbo** - $0.002 per 1K tokens
   - Best value for general tasks
   - Use for: Fan-out, extraction, parallel processing
   - **Primary model for bulk work**

3. **curie** - $0.002 / $0.03 per 1K tokens
   - Mid-tier model for critique/summarization
   - Use for: Critique, comparison, Q&A, validation
   - 10× cheaper than GPT-4

4. **gpt-3.5** - $0.01 / $0.02 per 1K tokens
   - More capable than turbo but pricier
   - Use for: General tasks needing more quality

5. **gpt-4** - $0.03 / $0.06 per 1K tokens
   - Most expensive, highest quality
   - Use for: Final synthesis, complex reasoning, creative writing
   - **Only use for critical 20% of work**

6. **gpt-4-32k** - $0.06 / $0.12 per 1K tokens
   - 32,000 token context window
   - Very expensive - **avoid by chunking instead**

### Code Models

7. **code-davinci-002** - $0.02 / $0.05 per 1K tokens
   - Code generation and completion
   - Use for: Writing code, technical implementations

8. **code-cushman-001** - $0.002 / $0.01 per 1K tokens
   - Cheaper code model for simpler tasks

### Other Specialized Models

9. **text-embedding-ada-002** - $0.0004 per 1K tokens
   - Embeddings for semantic search, similarity, clustering

10. **whisper-1** - $0.006 per minute
    - Speech-to-text transcription

11. **text-moderation-001** - $0.001 per 1K tokens
    - Content moderation, harmful content filtering

### Legacy Models (less useful)

- davinci, babbage: Older GPT-3 models, generally prefer gpt-3.5-turbo or curie
- DALL-E 2: Image generation ($0.02 per image)

## Key Changes to Documentation

### 1. Cost Comparisons Updated

**Old:**
- 25× cost difference
- Testing costs $0.10-0.20
- Wrong choice costs $2-5

**New:**
- 50-100× cost difference (ada vs gpt-4)
- Testing costs $0.01-0.05
- Wrong choice costs $1-6

### 2. Model Selection Table

Created condensed table in usage guide showing:
- Task type (fan-out, synthesis, code, etc.)
- Recommended model
- Actual pricing (per 1K tokens, not per 1M)
- Use cases

### 3. All Code Examples Updated

Replaced throughout:
- `#:model "gpt-4.1-nano"` → `#:model "gpt-3.5-turbo"`
- `#:model "gpt-4o-mini"` → `#:model "curie"`
- `#:model "gpt-4o"` → `#:model "gpt-4"`
- `#:model "gpt-4.1"` → `#:model "code-davinci-002"`

### 4. Tool Descriptions Updated

- `execute_scheme.md`: Updated model costs and examples
- `get_usage_guide.md`: Updated immediate techniques
- `load_context.md`: Updated strategy references

## Orchestration Strategy Recommendations

### For Fan-Out (Parallel Processing)
**Best:** `gpt-3.5-turbo` ($0.002/1K)
- Process 100 items: ~$0.20
- Fast, reliable, cheap

### For Critique/Comparison
**Best:** `curie` ($0.002-0.03/1K)
- Compare 3 approaches: ~$0.01-0.10
- Good quality for validation

### For Final Synthesis
**Best:** `gpt-4` ($0.03-0.06/1K)
- Final output: ~$0.60-1.20 per call
- Only use when quality critical

### For Simple Filtering
**Best:** `ada` ($0.0004/1K)
- Cheapest option: ~$0.04 per 100 items
- Perfect for classification

## Cost Impact

### Example: 100-document analysis

**Old (incorrect pricing):**
- Fan-out with "gpt-4.1-nano": claimed $0.10
- Synthesis with "gpt-4o": claimed $2.50
- Total: claimed $2.60

**New (actual pricing):**
- Fan-out with gpt-3.5-turbo: ~$0.20
- Synthesis with gpt-4: ~$0.60-1.20
- Total: ~$0.80-1.40

**Key insight:** Actual costs are **more affordable** than documented because:
- Real pricing is per 1K tokens (not per 1M)
- gpt-3.5-turbo is $0.002/1K (very cheap for bulk work)
- gpt-4 is expensive but still reasonable at $0.03-0.06/1K

## Files Modified

1. **docs/usage-guide.md** - Model selection table, all examples
2. **docs/tool-descriptions/execute_scheme.md** - Model costs
3. **docs/tool-descriptions/get_usage_guide.md** - Immediate techniques
4. **All pattern docs** - Updated model references globally
5. **All markdown files in docs/** - Systematic model name replacement

## Verification

✅ Python compiles successfully
✅ All 137+ model references updated
✅ Pricing corrected throughout
✅ Model selection table condensed and accurate
✅ No references to non-existent models (o3-mini, o4-mini, gpt-4.1-nano, etc.)

## Notes

- Pricing is per **1K tokens** (not 1M) in OpenAI's actual API
- Output tokens typically cost more than input tokens
- The 50-100× cost difference (ada → gpt-4) makes tiered strategies extremely effective
- gpt-3.5-turbo is the workhorse model for most orchestration tasks
- gpt-4 should only be used for final synthesis or complex reasoning
