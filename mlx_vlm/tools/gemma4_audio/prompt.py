def build_prompt(processor, model_config, user_text: str) -> str:
    """Build chat-template prompt with audio placeholder."""
    from mlx_vlm.prompt_utils import apply_chat_template
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    content = f"{audio_token}\n{user_text}"
    return apply_chat_template(processor, model_config, content, num_images=0)
