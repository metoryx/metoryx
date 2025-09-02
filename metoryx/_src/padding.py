from typing import Literal, Sequence

type PaddingLike = int | Sequence[int | tuple[int, int]] | Literal["SAME", "VALID"]


def canonicalize_padding(
    padding: PaddingLike, kernel_size: Sequence[int]
) -> Sequence[tuple[int, int]] | Literal["SAME", "VALID"]:
    if isinstance(padding, str):
        return padding
    elif isinstance(padding, int):
        return [(padding, padding) for _ in kernel_size]
    else:
        if len(padding) != len(kernel_size):
            raise ValueError(
                "If padding is a sequence, it must be of the same length as kernel_size"
            )
        return [(p, p) if isinstance(p, int) else p for p in padding]
