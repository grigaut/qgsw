"""Nox file."""

import nox


@nox.session()
@nox.session(venv_backend="conda")
def local(session: nox.Session) -> None:
    """Session from environment.yml."""
    session.run(
        "conda",
        "env",
        "update",
        "--prefix",
        session.virtualenv.location,
        "--file",
        "environment.yml",
    )
    session.install("pytest")
    session.install("pytest-sugar")
    session.install(".")
    session.run("pytest")
