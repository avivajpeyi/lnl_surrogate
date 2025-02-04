from enum import EnumMeta, auto
from typing import Union

from strenum import LowercaseStrEnum

from .gpflow_model import GPFlowModel
from .sklearn_gp_model import SklearnGPModel
from .sklearn_mlp_model import SklearnMlpModel
from .tf_dnn_model import TfDnnModel


class ModelType(LowercaseStrEnum):
    """Model type enum"""

    GPFlow = auto()
    SklearnGP = auto()
    SklearnMLP = auto()
    TfDnn = auto()

    @classmethod
    def list(cls):
        return [m.value for m in cls]

    @classmethod
    def from_str(cls, model_type: str) -> "ModelType":
        """Get model type from string"""
        model_type = model_type.lower()
        if model_type not in cls.list():
            raise ValueError(
                f"Model type {model_type} not recognised. Please choose from {cls.list()}"
            )
        return cls(model_type)


def load_model(model_type: Union[ModelType, str]):
    """Load a model from a string or ModelType"""
    if isinstance(model_type, str):
        model_type = ModelType.from_str(model_type)

    if model_type == ModelType.GPFlow:
        return GPFlowModel
    elif model_type == ModelType.SklearnGP:
        return SklearnGPModel
    elif model_type == ModelType.SklearnMLP:
        return SklearnMlpModel
    elif model_type == ModelType.TfDnn:
        return TfDnnModel
    else:
        raise ValueError(
            f"Model type {model_type} not recognised. Please choose from {ModelType.list()}"
        )
