"""Checks on the CONFIGURATION FILES (env examples, compose, Dockerfiles).

Why test config? Because config bugs are the #1 cause of 'it works on my laptop' failures in class.
One of these tests exists because a real bug was caught: an inline comment after an empty value
(`GROQ_MODEL=   # e.g. ...`) was read by Docker Compose as the VALUE of the variable.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLES = [ROOT / ".env.example", ROOT / "orchestrator" / ".env.example", ROOT / "mcp_server" / ".env.example"]


def compose(*names: str) -> dict:
    merged: dict = {}
    for name in names:
        doc = yaml.safe_load((ROOT / name).read_text())
        for service, spec in doc.get("services", {}).items():
            merged.setdefault(service, {})
            for key, value in spec.items():
                if isinstance(value, dict):
                    merged[service].setdefault(key, {}).update(value)
                else:
                    merged[service][key] = value
    return merged


# ---- .env.example files ---------------------------------------------------------------------

@pytest.mark.parametrize("path", ENV_EXAMPLES, ids=lambda p: str(p.relative_to(ROOT)))
def test_env_example_has_no_inline_comments_and_no_secrets(path):
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert "#" not in stripped.split("=", 1)[1], f"inline comment after a value (parsers disagree on these): {line!r}"
    values = dotenv_values(path)
    assert values, "file parsed to nothing"
    assert not values.get("GROQ_API_KEY"), "the example must not contain a key"
    assert not any(str(v).startswith("gsk_") for v in values.values())
    for key in ("GROQ_MODEL", "OLLAMA_MODEL"):
        assert values.get(key, "") == "", f"{key} must be empty so the user chooses a model on purpose"


def test_example_files_load_into_the_real_settings_classes():
    from app.config import Settings as OrchestratorSettings
    from settings import Settings as McpSettings

    orchestrator = OrchestratorSettings(_env_file=ROOT / "orchestrator" / ".env.example")
    assert orchestrator.llm_provider == "groq" and orchestrator.groq_model == "" and orchestrator.mcp_server_url.endswith("/mcp")
    mcp = McpSettings(_env_file=ROOT / "mcp_server" / ".env.example")
    assert mcp.ml_api_base_url == "http://localhost:8000" and "mcp-server:*" in mcp.allowed_hosts


def test_every_setting_in_the_service_env_examples_really_exists():
    """A typo like `GROK_API_KEY` would be silently ignored (extra='ignore'). Catch that."""
    from app.config import Settings as OrchestratorSettings
    from settings import Settings as McpSettings

    for path, cls in ((ROOT / "orchestrator" / ".env.example", OrchestratorSettings), (ROOT / "mcp_server" / ".env.example", McpSettings)):
        for key in dotenv_values(path):
            assert key.lower() in cls.model_fields, f"{path.name}: {key} is not a setting"


# ---- docker-compose ---------------------------------------------------------------------------

def test_compose_has_the_four_services_and_healthchecked_start_order():
    services = compose("docker-compose.yml")
    assert set(services) == {"frontend", "orchestrator", "mcp-server", "ml-api"}
    assert services["mcp-server"]["depends_on"]["ml-api"]["condition"] == "service_healthy"
    assert services["orchestrator"]["depends_on"]["mcp-server"]["condition"] == "service_healthy"
    assert services["frontend"]["depends_on"]["orchestrator"]["condition"] == "service_healthy"


def test_containers_reach_each_other_by_service_name_never_localhost():
    services = compose("docker-compose.yml")
    mcp_env, orch_env = services["mcp-server"]["environment"], services["orchestrator"]["environment"]
    assert mcp_env["ML_API_BASE_URL"] == "http://ml-api:8000"
    assert orch_env["MCP_SERVER_URL"] == "http://mcp-server:8001/mcp"
    assert mcp_env["MCP_HOST"] == "0.0.0.0"
    assert "mcp-server:*" in mcp_env["MCP_ALLOWED_HOSTS"]
    for service in ("mcp-server", "orchestrator"):
        for key, value in services[service]["environment"].items():
            if key in ("ML_API_BASE_URL", "MCP_SERVER_URL"):
                assert "localhost" not in str(value) and "127.0.0.1" not in str(value)


def test_compose_never_contains_a_secret_and_passes_the_key_only_to_the_orchestrator():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "gsk_" not in text
    services = compose("docker-compose.yml")
    holders = [name for name, spec in services.items() if "GROQ_API_KEY" in (spec.get("environment") or {})]
    assert holders == ["orchestrator"]
    assert "GROQ" not in str(services["frontend"]) and "GROQ" not in str(services["mcp-server"])


def test_frontend_receives_only_a_browser_facing_url_at_build_time():
    args = compose("docker-compose.yml")["frontend"]["build"]["args"]
    assert set(args) == {"VITE_API_BASE_URL"} and "localhost" in args["VITE_API_BASE_URL"]


def test_ollama_override_points_the_orchestrator_at_the_service_name():
    services = compose("docker-compose.yml", "docker-compose.ollama.yml")
    assert services["orchestrator"]["environment"]["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert services["orchestrator"]["depends_on"]["ollama"]["condition"] == "service_healthy"
    assert "ollama" not in compose("docker-compose.yml")  # the base file does not need it


# ---- Dockerfiles ------------------------------------------------------------------------------

@pytest.mark.parametrize("folder", ["mcp_server", "orchestrator", "frontend"])
def test_dockerfiles_run_unprivileged_with_a_healthcheck_and_copy_no_secrets(folder):
    text = (ROOT / folder / "Dockerfile").read_text()
    assert "HEALTHCHECK" in text
    assert re.search(r"^USER\s+\S+|nginx-unprivileged", text, re.M), "container should not run as root"
    assert not re.search(r"COPY\s+.*\.env(?!\.example)", text)
    ignore = (ROOT / folder / ".dockerignore").read_text() if (ROOT / folder / ".dockerignore").exists() else ""
    assert ".env" in ignore


def test_python_dockerfiles_copy_everything_the_service_imports():
    """The mcp_server image copies *.py and tools/ : make sure every top-level module is a .py there."""
    dockerfile = (ROOT / "mcp_server" / "Dockerfile").read_text()
    assert "COPY *.py" in dockerfile and "COPY tools/" in dockerfile
    modules = {p.name for p in (ROOT / "mcp_server").iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert modules <= {"tools"}


# ---- the browser must never hold a secret or talk to the LLM ------------------------------------------

def test_frontend_source_never_mentions_llm_providers_or_keys():
    for path in (ROOT / "frontend" / "src").rglob("*"):
        if path.is_file() and "test" not in path.parts and path.suffix in {".ts", ".tsx"}:
            text = path.read_text().lower()
            for forbidden in ("groq.com", "11434", "api_key", "authorization", "bearer "):
                assert forbidden not in text, f"{path.name} mentions {forbidden}"


# ---- documentation ------------------------------------------------------------------------------

def test_relative_links_in_the_docs_point_at_files_that_exist():
    link = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")
    docs = [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), ROOT / "ml_api" / "README.md"]
    for doc in docs:
        for target in link.findall(doc.read_text()):
            path = (doc.parent / target.split("#")[0]).resolve()
            assert path.exists(), f"{doc.relative_to(ROOT)} links to missing {target}"


def test_files_the_docs_tell_students_to_run_exist():
    for name in ("scripts/smoke_test.sh", "scripts/smoke_test.ps1", "scripts/fake_ollama.py", "docker-compose.ollama.yml",
                 "requirements-dev.txt", ".env.example", "orchestrator/.env.example", "mcp_server/.env.example", "frontend/.env.example"):
        assert (ROOT / name).exists(), name
