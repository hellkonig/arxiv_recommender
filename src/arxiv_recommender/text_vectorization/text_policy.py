"""Extensible paper text construction for embedding inputs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import JsonValue


class PaperTextInput(Protocol):
    """Paper fields required by the current text-policy contract."""

    title: str
    abstract: str


class PaperTextPolicy(ABC):
    """Construct model input text and describe the applied policy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable policy identifier."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the semantic version of the policy behavior."""

    @property
    @abstractmethod
    def config(self) -> dict[str, JsonValue]:
        """Return score-affecting policy configuration."""

    @abstractmethod
    def build_text(self, paper: PaperTextInput) -> str:
        """Construct embedding input from paper metadata."""


@dataclass(frozen=True)
class TitleAbstractTextPolicy(PaperTextPolicy):
    """Join the title and abstract with a configurable separator."""

    NAME: ClassVar[str] = "title_abstract"
    VERSION: ClassVar[str] = "1.0.0"

    separator: str = " "

    @property
    def name(self) -> str:
        """Return the stable policy identifier."""
        return self.NAME

    @property
    def version(self) -> str:
        """Return the semantic version of this policy behavior."""
        return self.VERSION

    @property
    def config(self) -> dict[str, JsonValue]:
        """Return configuration needed to reproduce the policy."""
        return {"separator": self.separator}

    def build_text(self, paper: PaperTextInput) -> str:
        """Construct text with the title before the abstract."""
        return f"{paper.title}{self.separator}{paper.abstract}"
