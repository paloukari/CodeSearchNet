from typing import Any, Dict, Optional, Type

from .model import Model
from encoders import GraphEncoder, Encoder


class GraphModel(Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        model_hypers = {}
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 code_encoder_type: Type[GraphEncoder],
                 query_encoder_type: Type[Encoder],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None,
                 ):
        super().__init__(
            hyperparameters,
            code_encoder_type=code_encoder_type,
            query_encoder_type=query_encoder_type,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)
