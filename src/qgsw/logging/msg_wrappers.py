"""Message wrappers."""


def surround(msg: str, char: str = "#") -> str:
    """Surround msg with any character.

    Args:
        msg (str): Message to surround.
        char (str, optional): Character for surrounding. Defaults to "#".

    Raises:
        ValueError: If char is not a single character.

    Returns:
        str: Surrounded message.
    """
    if len(char) != 1:
        msg = "The fill character must be exactly one character long."
        raise ValueError(msg)
    msg_parts = msg.split("\n")
    max_len = max(len(p) for p in msg_parts)
    msg_ = [f"{char} {p.center(max_len, ' ')} {char}" for p in msg_parts]
    outer = "".join([char] * (max_len + 4))
    inner = f"{char} " + "".join([" "] * (max_len)) + f" {char}"
    return "\n".join([outer, inner, *msg_, inner, outer])
