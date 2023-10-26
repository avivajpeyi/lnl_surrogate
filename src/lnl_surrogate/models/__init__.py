from .tf_dnn_model import TfDnnModel
from .gpflow_model import GPFlowModel
from .sklearn_mlp_model import SklearnMlpModel
from .sklearn_gp_model import SklearnGPModel

from enum import Enum


class ModelType(Enum):
    """Model type enum"""
    GPFlow = "gpflow"
    SklearnGP = "sklearn_gp"
    SklearnMLP = "sklearn_mlp"
    TfDnn = "tf_dnn"

    @staticmethod
    def list():
        """List of model types"""
        return [m.value for m in ModelType]

    @classmethod
    def from_str(self, model_str: str):
        """Get model class from string (case-insensitive)"""
        model_str = model_str.lower()
        if model_str == self.GPFlow.__name__.lower():
            return GPFlowModel
        elif model_str == self.SklearnGP.__name__.lower():
            return SklearnGPModel
        elif model_str == self.SklearnMLP.__name__.lower():
            return SklearnMlpModel
        elif model_str == self.TfDnn.__name__.lower():
            return TfDnnModel
        else:
            raise ValueError(f"Model type {model_str} not recognised. Please choose from {self.list}")

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
