"""Re-exports `agents.stages.gold.service`'s public API at the package level, so callers can
do `from agents.stages import gold` and then `gold.current_architecture_diagram(...)`,
`gold.persist_entity_version(...)`, etc. — the exact same call-site shape the old flat
`agents/gold_service.py` module gave `from agents import gold_service` callers, now backed by
this package instead of a single file."""

from agents.stages.gold.service import *  # noqa: F401, F403
