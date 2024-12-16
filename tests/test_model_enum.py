import pytest

from lnl_surrogate.models import ModelType


def test_model_enum():
    assert ModelType.from_str("gpflow") == ModelType.GPFlow
    # assert raises ValueError
    with pytest.raises(ValueError):
        ModelType.from_str("gpfloww")
    assert "gpflow" in ModelType.list()
