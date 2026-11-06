"""Explicit context cropping; reject shape mismatches instead of broadcasting loss."""


def crop_prediction(prediction, target, context=0):
    if context < 0:
        raise ValueError("Context cannot be negative")
    if prediction.shape[:-2] != target.shape[:-2]:
        raise ValueError("Prediction batch/channel dimensions must match target")
    expected = tuple(size + 2 * context for size in target.shape[-2:])
    if prediction.shape[-2:] != expected:
        raise ValueError(f"Expected prediction spatial shape {expected}, got {prediction.shape[-2:]}")
    return prediction[..., context:-context, context:-context] if context else prediction
