"""Keyword-based classifier for equipment values: standard=2, optional=1, not present=0.

This is the "matcher" baseline — fast and free, but only handles values that
spell out their meaning plainly. Whatever it can't confidently classify is
left to the LLM in classify_equipment.py.
"""
import re

# Equipment columns whose values describe standard/optional/not-present status
# for a single piece of equipment (fits the 0/1/2 scale). Excludes location,
# type, and NHTSA-evaluation-rating columns (e.g. *_LOC, *_TYPE,
# NHTSA_*_EVALUATION, HEAD_RESTRAINT_IND's seat-position codes) which don't.
CORE_FEATURES = [
    "UPPER_BELT_ANCHORAGE", "SEAT_BELT_PRETENSIONER", "LOAD_LIMITERS",
    "FRNT_BELT_INDICATOR", "REAR_BELT_INDICATOR",
    "HEAD_SAB", "TORSO_SAB", "PELVIS_SAB", "KNEE_BOLSTERS",
    "ADL", "BETI", "ARS",
    "BLIND_SPOT_DETECTION", "DAY_RUN_LIGHTS", "ADAPTIVE_CRUISE_CONTROL", "ABS",
    "AUTO_CRASH_NOTIFICATION", "ANTI_THEFT_DEVICE",
    "FRNT_COLLISION_WARNING", "LANE_DEPARTURE_WARNING",
    "CRASH_IMMINENT_BRAKE", "DYNAMIC_BRAKE_SUPPORT", "NHTSA_ESC",
    "BACKUP_CAMERA",
]

STANDARD_TOKENS = {"standard", "std", "std.", "s"}
# Bare "S" is unambiguous ("Standard"), but bare "A" is not — validated against
# the LLM at 97.5% agreement overall, with the one disagreement being "A"
# (rule guessed standard, LLM read it as available/optional). Left for the LLM.
OPTIONAL_TOKENS = {"optional", "opt", "o", "avl", "available", "avail"}
NOT_PRESENT_TOKENS = {"not available", "none", "n", "no"}


def classify_rule_based(raw_value):
    """Return (level, confidence) or None if the value isn't clearly handled.

    level: 2 = standard, 1 = optional/available, 0 = not present.
    """
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if not value:
        return None

    # Strip a single trailing parenthetical/bracket like "(57%)" or "[23%]"
    # before matching, since it's a qualifier on an otherwise-clear token.
    stripped = re.sub(r"\s*[\(\[][^()\[\]]*[\)\]]\s*$", "", value).strip()

    if stripped in STANDARD_TOKENS:
        return (2, "high")
    if stripped in OPTIONAL_TOKENS:
        return (1, "high")
    if stripped in NOT_PRESENT_TOKENS:
        return (0, "high")

    return None
