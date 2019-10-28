from typing import Any, Dict, Optional

from encoders import RGCN_Encoder, ConvSelfAttentionEncoder
from models import GraphModel


class RGCN2TransformerModel(GraphModel):

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        hypers.update({f'code_{key}': value
                       for key, value in RGCN_Encoder.get_default_hyperparameters().items()})
        hypers.update({f'query_{key}': value
                       for key, value in ConvSelfAttentionEncoder.get_default_hyperparameters().items()})
        model_hypers = {
            'learning_rate': 5e-4,
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'batch_size': 200,
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            code_encoder_type=RGCN_Encoder,
            query_encoder_type=ConvSelfAttentionEncoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)
