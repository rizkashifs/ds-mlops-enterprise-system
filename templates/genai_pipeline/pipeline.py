"""Template: GenAI / LLM pipeline.

Copy this for use cases that involve text, reasoning, or extraction with an LLM.
Replace all TODO items with your implementation.

Before using this template, confirm an LLM is the right tool.
See docs/decision-frameworks.md §1 (ML vs LLM).

Use this when:
  - Input is unstructured text
  - Task involves reasoning, extraction, or generation
  - No labeled training data exists (or few-shot is sufficient)

Consider ML instead when:
  - You have >1,000 labeled examples
  - Latency < 200ms is required
  - Cost at scale is a constraint
  - Auditability / reproducibility is critical

This template uses the Anthropic API (Claude).
Install: pip install anthropic
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0     # 0 = deterministic; increase for creative tasks
    system_prompt: str = ""


@dataclass
class ExtractionResult:
    input_text: str
    output: Dict[str, Any]
    model: str
    usage: Dict[str, int]


def extract_with_llm(text: str, config: LLMConfig) -> ExtractionResult:
    """TODO: replace this with your actual extraction/generation task.

    This example extracts structured fields from free text.
    Adapt the system prompt and output parsing for your use case.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # TODO: design your system prompt for your specific task
    system = config.system_prompt or (
        "Extract the requested information from the provided text. "
        "Return a JSON object. If a field cannot be determined, use null."
    )

    # TODO: design your user prompt / task description
    user_prompt = f"""
Extract the following from the text below:
- sentiment: positive/negative/neutral
- key_topics: list of main topics (max 3)
- action_required: true/false

Text: {text}

Return JSON only.
"""

    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )

    import json
    try:
        output = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        output = {"raw": message.content[0].text}

    return ExtractionResult(
        input_text=text,
        output=output,
        model=config.model,
        usage={"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
    )


# ---------------------------------------------------------------------------
# Governance note for LLM pipelines
# ---------------------------------------------------------------------------
#
# LLM outputs are non-deterministic. For governed use cases:
#   - Log every input + output + model version (for auditability)
#   - Set temperature=0 for extraction tasks requiring consistency
#   - Define a human review step for high-stakes outputs
#   - Version your prompts (treat prompt changes like model version bumps)
#   - Monitor output quality with a sample of human-reviewed outputs
#
# There is no MLflow model artifact for LLM pipelines —
# but you should still log: model name, model version, prompt hash,
# input/output counts, and any post-processing logic version.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    config = LLMConfig()
    sample_text = "The customer called to complain about a billing error. They were frustrated but agreed to wait for a resolution."
    result = extract_with_llm(sample_text, config)
    print(f"Output: {result.output}")
    print(f"Tokens used: {result.usage}")
