"""The fixed list of agents the project speaks as.

The agent is always chosen from this list. There is deliberately no way to condition
the model on an arbitrary uploaded voice (see the legal note in the README).
"""

AGENT_NAMES = (
    "Astra",
    "Breach",
    "Brimstone",
    "Chamber",
    "Clove",
    "Cypher",
    "Deadlock",
    "Fade",
    "Gekko",
    "Harbor",
    "Iso",
    "Jett",
    "KAY/O",
    "Killjoy",
    "Miks",
    "Neon",
    "Omen",
    "Phoenix",
    "Raze",
    "Reyna",
    "Sage",
    "Skye",
    "Sova",
    "Tejo",
    "Veto",
    "Viper",
    "Vyse",
    "Waylay",
    "Yoru",
)


def slugify(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


AGENTS = {slugify(name): name for name in AGENT_NAMES}


def require_agent(slug: str) -> str:
    """Returns the slug when it is a known agent, otherwise raises ValueError."""
    if slug not in AGENTS:
        raise ValueError(f"Unknown agent '{slug}'. Known agents: {', '.join(sorted(AGENTS))}")
    return slug
