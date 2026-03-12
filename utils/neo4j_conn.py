"""
utils/neo4j_conn.py
-------------------
Shared Neo4j driver factory.

Reads connection details from environment variables (or an optional .env file).
All pipeline scripts import get_driver() from here — connection config is never
duplicated across step files.

Environment variables required:
    NEO4J_URI       e.g. bolt://localhost:7687
    NEO4J_USER      e.g. neo4j
    NEO4J_PASSWORD  (never log this value — see CLAUDE.md Hard Rule #4)
"""

import os
import logging
from typing import Optional
from neo4j import GraphDatabase, Driver

log = logging.getLogger(__name__)


def _load_dotenv(env_path: Optional[str] = None) -> None:
    """
    Optionally load a .env file.  Silently skips if python-dotenv is not
    installed or the file does not exist — environment variables already set
    in the shell take precedence over the .env file.
    """
    try:
        from dotenv import load_dotenv
        path = env_path or ".env"
        if os.path.exists(path):
            load_dotenv(path, override=False)  # shell vars win
            log.debug("Loaded env from %s", path)
    except ImportError:
        pass  # python-dotenv optional


def get_driver(env_path: Optional[str] = None) -> Driver:
    """
    Return an authenticated Neo4j Driver.

    Usage:
        driver = get_driver()
        with driver.session() as session:
            session.run(...)
        driver.close()

    Or use it as a context manager:
        with get_driver() as driver:
            ...

    Raises:
        EnvironmentError  if any required variable is missing
        neo4j.exceptions.ServiceUnavailable  if Neo4j is not reachable
    """
    _load_dotenv(env_path)

    uri      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.environ.get("NEO4J_USER",     "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        raise EnvironmentError(
            "NEO4J_PASSWORD is not set. "
            "Add it to your .env file or export it in your shell."
        )

    # Log URI and user but NEVER the password (CLAUDE.md Hard Rule #4)
    log.info("Connecting to Neo4j at %s as %s", uri, user)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()          # fast fail if unreachable
    log.info("Neo4j connection verified.")
    return driver
