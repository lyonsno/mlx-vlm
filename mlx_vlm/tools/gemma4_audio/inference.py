from typing import Callable, Iterator

import numpy as np


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
    from mlx_vlm.server.responses_state import ThinkingStreamState

    prompt = prompt_builder(user_prompt)
    state = ThinkingStreamState()
    acc = ""

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
            chunk = token.text if hasattr(token, "text") else str(token)
            delta = state.feed(chunk)
            if delta.content:
                acc += delta.content
                yield acc
    except Exception as e:
        yield f"[error] {e}"
