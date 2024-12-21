"""Nox file."""

import nox


@nox.session()
@nox.session(venv_backend="conda")
def local(session: nox.Session) -> None:
    """Session from environment-local.yml."""
    session.run(
        "conda",
        "env",
        "update",
        "--prefix",
        session.virtualenv.location,
        "--file",
        "environment-local.yml",
    )
    session.install("pytest")
    session.install("pytest-sugar")
    session.install(".")
    session.run("pytest")


@nox.session(venv_backend="conda")
def g5000(session: nox.Session) -> None:
    """Session from environment-g5000.yml."""
    session.run(
        "conda",
        "env",
        "update",
        "--prefix",
        session.virtualenv.location,
        "--file",
        "environment-g5000.yml",
    )
    session.install("pytest")
    session.install("pytest-sugar")
    session.install(".")
    session.run("pytest")
