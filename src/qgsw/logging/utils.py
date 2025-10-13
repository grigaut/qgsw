"""Message wrappers."""

from __future__ import annotations

from typing import Literal


def assert_char(char: str) -> None:
    """Assert that a string is a character.

    Args:
        char (str): Character.

    Raises:
        ValueError: If char is not a single character.
    """
    if len(char) != 1:
        msg = "The fill character must be exactly one character long."
        raise ValueError(msg)


def pad(msg: str, char: str = "#", width: int = 1) -> str:
    """Horizontally pad a message.

    Args:
        msg (str): Message to pad.
        char (str, optional): Padding character. Defaults to "#".
        width (int, optional): Padding width. Defaults to 1.

    Returns:
        str: Padded message.
    """
    assert_char(char)
    msg_parts = msg.split("\n")
    max_len = max(len(p) for p in msg_parts)
    msg_ = [p.center(max_len, " ") for p in msg_parts]
    pad_seq = "".join([char] * width)
    msg = [pad_seq + p + pad_seq for p in msg_]
    return "\n".join(msg)


def banner(msg: str, char: str = "=") -> str:
    """Add a banner to a message.

    Args:
        msg (str): Message to banner.
        char (str, optional): Character for banner. Defaults to "=".

    Returns:
        str: Message within banner.
    """
    assert_char(char)
    msg_parts = msg.split("\n")
    max_len = max(len(p) for p in msg_parts)
    msg_ = [p.center(max_len, " ") for p in msg_parts]
    outer = "".join([char] * (max_len))
    return "\n".join([outer, *msg_, outer])


box_styles = {
    "-": ("─", "│", "┌", "┐", "└", "┘"),
    "round": ("─", "│", "╭", "╮", "╰", "╯"),
    "=": ("═", "║", "╔", "╗", "╚", "╝"),
}

BoxStyles = Literal["-", "round", "="]


def box(
    msg: str,
    *,
    style: BoxStyles = "=",
    char: str | None = None,
) -> str:
    """Draw a box around a message.

    Args:
        msg (str): Message to draw a box around.
        style (BoxStyles, optional): Box style. Ignored if char is not None.
            Defaults to "=".
        char (str | None, optional): Char to use to draw the border.
            Take precedence over 'style'.Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        str: _description_
    """
    if char is None and style not in box_styles:
        msg = f"Available styles are {', '.join(list(box_styles.keys()))}"
        raise ValueError(msg)
    if char is not None:
        assert_char(char)
        h = v = tl = tr = bl = br = char
    else:
        h, v, tl, tr, bl, br = box_styles[style]
    blank_banner = banner(msg, " ")
    msg_pad = pad(blank_banner, " ")
    msg_parts = msg_pad.split("\n")
    max_len = max(len(p) for p in msg_parts)
    top = tl + "".join([h] * max_len) + tr
    bot = bl + "".join([h] * max_len) + br
    msg_pad_ = pad(msg_pad, v)
    return "\n".join([top, *msg_pad_.split("\n"), bot])


def step(current: int, total: int | None = None) -> str:
    """Create a string representing a step.

    Args:
        current (int): Current step.
        total (int | None, optional): Total steps. Defaults to None.

    Returns:
        str: current / total.
    """
    c_str = str(current)
    t_str = "?" * max(3, len(c_str)) if total is None else str(total)
    return c_str.zfill(len(t_str)) + "/" + t_str
