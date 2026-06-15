import re
from typing import Callable, Iterator

import numpy as np

# Matches complete and partial thinking blocks
_THINKING_RE = re.compile(
    r'<\|channel>thought\n.*?<channel\|>'   # complete block
    r'|<\|channel>thought\n.*$'              # incomplete trailing block
    r'|<\|channel>thought.*$'                # partial tag at end
    r'|<\|channel>.*$',                      # very partial tag at end
    re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove all <|channel>thought...<channel|> blocks from text."""
    return _THINKING_RE.sub('', text).strip()


def run_inference(
    model,
    processor,
    audio: np.ndarray,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    prompt_builder: Callable[[str], str],
) -> Iterator[str]:
    """Yield accumulated visible response text as tokens arrive.

    Strips <|channel>thought...</channel|> thinking tokens automatically.
    Yields the full accumulated string on each update (suitable for Gradio streaming).
    """
    from mlx_vlm.generate import stream_generate

    prompt = prompt_builder(user_prompt)
    raw_acc = ""
    prev_clean = ""

    try:
        for token in stream_generate(
            model,
            processor,
            prompt=prompt,
            audio=[audio],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        ):
            # stream_generate yields per-token deltas via .text
            delta = token.text if hasattr(token, "text") else str(token)
            if not delta:
                continue
            raw_acc += delta
            clean = _strip_thinking(raw_acc)
            if clean and clean != prev_clean:
                prev_clean = clean
                yield clean
    except Exception as e:
        yield f"[error] {e}"
