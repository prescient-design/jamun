# https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/optimization.py#L98C1-L101C117
def linear_warmup_linear_decay_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
    )


def linear_warmup_plateau_lr_lambda(
    current_step: int, *, num_warmup_steps: int, start_factor: float = 0.0, end_factor: float = 1.0
):
    f = min(1.0, current_step / num_warmup_steps)
    return start_factor * (1 - f) + f * end_factor


def linear(current_step: int, *, start_factor: float = 0.0, slope: float = 1e-6):
    return max(0.0, start_factor + current_step * slope)
