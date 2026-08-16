import pytest
from pydantic import ValidationError

from arxiv_recommender.provenance import ModelProvenance


def test_model_provenance_normalizes_identity() -> None:
    provenance = ModelProvenance(name=" model ", version=" 1.0.0 ")

    assert provenance.name == "model"
    assert provenance.version == "1.0.0"


@pytest.mark.parametrize("field_name", ["name", "version"])
def test_model_provenance_rejects_blank_identity(field_name: str) -> None:
    values = {"name": "model", "version": "1.0.0", field_name: "   "}

    with pytest.raises(ValidationError, match="must not be blank"):
        ModelProvenance(name=values["name"], version=values["version"])
