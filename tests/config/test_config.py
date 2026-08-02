"""
test_config.py

Exercises ConfigLoader / Config parsing:

  1. valid_full.yaml           -> should load cleanly; returns the Config object.
  2. invalid_llm_ref.yaml      -> agent references an unknown llm -> ValueError.
  3. invalid_prompt_conflict.yaml -> agent sets both prompt & prompt_file -> ValueError.
  4. missing env var           -> ${GEMINI_API_KEY} unset -> ValueError.
  5. self-peer / unknown-peer  -> built in-memory (one-liners, no YAML needed) -> ValueError.

Run with:  python test_config.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config import Config, ConfigLoader

BASE_DIR = Path(__file__).parent
CONFIGS_DIR = BASE_DIR / "configs"

FAILURES = 0


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  [FAIL] {msg}")


def test_valid_config() -> Config:
    print("\n1) valid_full.yaml — should load cleanly")
    os.environ["GEMINI_API_KEY"] = "test-gemini-key-123"

    cfg = ConfigLoader.load(CONFIGS_DIR / "valid_full.yaml")

    print("\n  --- Config object (repr) ---")
    print(" ", cfg)

    print("\n  --- Config object (as JSON) ---")
    print(cfg.model_dump_json(indent=2))

    if isinstance(cfg, Config):
        _pass("ConfigLoader.load returned a Config instance")
    else:
        _fail(f"expected Config instance, got {type(cfg)}")

    if cfg.llms["gemini_default"].api_key == "test-gemini-key-123":
        _pass("${GEMINI_API_KEY} was resolved from the environment")
    else:
        _fail(f"api_key not resolved correctly: {cfg.llms['gemini_default'].api_key!r}")

    if cfg.agents["researcher"].name == "researcher":
        _pass("agent name injected from YAML key ('researcher')")
    else:
        _fail(f"agent name not injected correctly: {cfg.agents['researcher'].name!r}")

    researcher_prompt = cfg.agents["researcher"].resolve_prompt()
    if researcher_prompt and "careful researcher" in researcher_prompt:
        _pass("inline `prompt` resolved correctly")
    else:
        _fail(f"inline prompt did not resolve as expected: {researcher_prompt!r}")

    writer_prompt = cfg.agents["writer"].resolve_prompt()
    if writer_prompt and "polished, engaging copy" in writer_prompt:
        _pass("`prompt_file` resolved by reading the file from disk")
    else:
        _fail(f"prompt_file did not resolve as expected: {writer_prompt!r}")

    if cfg.agents["researcher"].peers == ["writer"] and cfg.agents["writer"].peers == ["researcher"]:
        _pass("peer lists parsed correctly")
    else:
        _fail("peer lists did not parse as expected")

    if [s.name for s in cfg.mcp_servers] == ["filesystem", "search"]:
        _pass("mcp_servers parsed correctly")
    else:
        _fail(f"mcp_servers not parsed as expected: {cfg.mcp_servers!r}")

    return cfg


def test_unknown_llm_reference() -> None:
    print("\n2) invalid_llm_ref.yaml — agent references an undefined llm, should raise")
    os.environ["GEMINI_API_KEY"] = "test-gemini-key-123"
    try:
        ConfigLoader.load(CONFIGS_DIR / "invalid_llm_ref.yaml")
        _fail("expected ValueError, but load() succeeded")
    except ValueError as e:
        if "unknown llm" in str(e):
            _pass(f"raised ValueError as expected ({e.__class__.__name__})")
        else:
            _fail(f"raised ValueError but with unexpected message: {e}")
    except Exception as e:
        _fail(f"raised wrong exception type {type(e).__name__}: {e}")


def test_prompt_and_prompt_file_conflict() -> None:
    print("\n3) invalid_prompt_conflict.yaml — agent sets both prompt and prompt_file, should raise")
    os.environ["GEMINI_API_KEY"] = "test-gemini-key-123"
    try:
        ConfigLoader.load(CONFIGS_DIR / "invalid_prompt_conflict.yaml")
        _fail("expected ValueError, but load() succeeded")
    except ValueError as e:
        if "set only one of" in str(e):
            _pass(f"raised ValueError as expected ({e.__class__.__name__})")
        else:
            _fail(f"raised ValueError but with unexpected message: {e}")
    except Exception as e:
        _fail(f"raised wrong exception type {type(e).__name__}: {e}")


def test_missing_env_var() -> None:
    print("\n4) missing env var — ${GEMINI_API_KEY} unset, should raise")
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        ConfigLoader.load(CONFIGS_DIR / "valid_full.yaml")
        _fail("expected ValueError, but load() succeeded")
    except ValueError as e:
        if "GEMINI_API_KEY" in str(e):
            _pass(f"raised ValueError as expected: {e}")
        else:
            _fail(f"raised ValueError but with unexpected message: {e}")
    except Exception as e:
        _fail(f"raised wrong exception type {type(e).__name__}: {e}")
    finally:
        os.environ["GEMINI_API_KEY"] = "test-gemini-key-123"


def test_self_peer_and_unknown_peer_in_memory() -> None:
    print("\n5) in-memory checks: self-peer and unknown-peer should both raise")

    self_peer_dict = {
        "llms": {"gemini_default": {"provider": "gemini", "model": "gemini-2.5-pro", "api_key": "x"}},
        "agents": {
            "solo": {"llm": "gemini_default", "peers": ["solo"]},
        },
    }
    try:
        Config.model_validate(self_peer_dict)
        _fail("expected ValueError for self-peer, but validation succeeded")
    except ValueError as e:
        if "cannot list itself as a peer" in str(e):
            _pass(f"self-peer raised ValueError as expected")
        else:
            _fail(f"self-peer raised ValueError with unexpected message: {e}")

    unknown_peer_dict = {
        "llms": {"gemini_default": {"provider": "gemini", "model": "gemini-2.5-pro", "api_key": "x"}},
        "agents": {
            "solo": {"llm": "gemini_default", "peers": ["ghost"]},
        },
    }
    try:
        Config.model_validate(unknown_peer_dict)
        _fail("expected ValueError for unknown peer, but validation succeeded")
    except ValueError as e:
        if "unknown peer" in str(e):
            _pass(f"unknown-peer raised ValueError as expected")
        else:
            _fail(f"unknown-peer raised ValueError with unexpected message: {e}")


def main() -> None:
    cfg = test_valid_config()
    test_unknown_llm_reference()
    test_prompt_and_prompt_file_conflict()
    test_missing_env_var()
    test_self_peer_and_unknown_peer_in_memory()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{FAILURES} check(s) FAILED")
        sys.exit(1)

    print("All checks PASSED")


if __name__ == "__main__":
    main()