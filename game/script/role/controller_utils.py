"""Player controller type helpers (Human / RL / Bot)."""

from __future__ import annotations

from typing import Literal

PlayerController = Literal["Human", "RL", "Bot"]
CONTROLLER_CHOICES: tuple[str, ...] = ("Human", "RL", "Bot")


def infer_controller_from_legacy_name(text: str) -> PlayerController:
    """Infer controller type from legacy display names (e.g. Human_player1)."""
    lowered = str(text or "").lower()
    if "human" in lowered:
        return "Human"
    if "bot" in lowered:
        return "Bot"
    if "rl" in lowered:
        return "RL"
    return "Bot"


def normalize_controller(controller: str | None) -> PlayerController:
    """Return explicit controller field; default to Bot when unset."""
    if controller in CONTROLLER_CHOICES:
        return controller
    return "Bot"


def parse_controller_override(value: str) -> PlayerController:
    """Parse training preset controller override (``RL`` / legacy ``RL_player1``)."""
    text = str(value or "").strip()
    if text in CONTROLLER_CHOICES:
        return text
    return infer_controller_from_legacy_name(text)


def normalize_player_controller_overrides(values: list[str]) -> list[PlayerController]:
    return [parse_controller_override(v) for v in values or []]
