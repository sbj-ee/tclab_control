"""Plant connection helper.

Returns a TCLab instance — the real board if available, otherwise the built-in
TCLabModel simulator. This lets every experiment run with no hardware connected
(``use_sim=True``) and then move to the real device by flipping one flag.
"""

from __future__ import annotations


def connect(use_sim: bool = False):
    """Return a TCLab-like context manager.

    Args:
        use_sim: force the offline TCLabModel simulator. If False, tries the real
            board and is the caller's responsibility to have it connected.

    Returns:
        An object usable as ``with connect() as lab: lab.T1 / lab.Q1(...)``.

    Raises:
        ImportError: if the optional ``tclab`` dependency isn't installed
            (install with ``pip install "tclab_control[hardware]"``).
    """
    try:
        import tclab
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "the 'tclab' package is required; install with "
            'pip install "tclab_control[hardware]"'
        ) from exc

    return tclab.TCLabModel() if use_sim else tclab.TCLab()
