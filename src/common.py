from typing import TypedDict


class InputItem(TypedDict):
    """An item of input data."""

    id: str
    syllogism: str


class OutputItem(TypedDict):
    """An item of output data."""

    id: str
    validity: bool
