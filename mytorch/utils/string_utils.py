def format_float(v: float) -> str:
    """Display a float in scientific notation only if it's too small or large.
    """
    if 1e-3 < abs(v) < 1e3:
        return f"{v:.5g}"
    return f"{v:.4e}"
