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
    prev_raw = ""

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
            # stream_generate yields accumulated text, not deltas.
            # Extract the delta for ThinkingStreamState which expects incremental input.
            raw = token.text if hasattr(token, "text") else str(token)
            delta_raw = raw[len(prev_raw):]
            prev_raw = raw

            if not delta_raw:
                continue

            delta = state.feed(delta_raw)
            if delta.content:
                acc += delta.content
                yield acc
    except Exception as e:
        yield f"[error] {e}"
