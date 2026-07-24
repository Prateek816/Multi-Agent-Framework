from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class AgentConfig(BaseModel):
    """
    One entry under `agents:` in the framework's YAML config.
    Pure data — no resolution logic here. AgentFactory is what
    turns this into a live BaseAgent instance.
    """

    name: str
    type: Optional[str] = None          # plugin class name; None = default ReactAgent
    llm: str                            # key into LLMRegistry
    tools: list[str] = Field(default_factory=list)  # names / wildcards / bundle names
    memory: Optional[str] = None        # scope name; defaults to `name` if absent

    prompt: Optional[str] = None        # inline prompt string
    prompt_file: Optional[str] = None   # path to a prompt file, alternative to `prompt`

    extra: dict[str, Any] = Field(default_factory=dict)  # passed through to from_config()

    @model_validator(mode="after")
    def _validate_prompt_source(self) -> "AgentConfig":
        if self.prompt and self.prompt_file:
            raise ValueError(
                f"Agent '{self.name}': set only one of `prompt` or `prompt_file`, not both."
            )
        return self

    def resolve_prompt(self) -> Optional[str]:
        """Returns the final prompt string, reading prompt_file from disk if that's what was set."""
        if self.prompt is not None:
            return self.prompt
        if self.prompt_file is not None:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        return None