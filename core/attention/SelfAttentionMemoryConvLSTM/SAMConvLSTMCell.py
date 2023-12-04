import sys
from typing import Optional, Tuple, Union

import torch

sys.path.append(".")
from core.utils.constants import DEVICE, WeightsInitializer  # noqa: E402
from core.attention.SelfAttentionConvLSTM.BaseConvLSTMCell import BaseConvLSTMCell  # noqa: E402
from core.attention.SelfAttentionMemoryConvLSTM.SelfAttentionMemory import (  # noqa: E402
    SelfAttentionMemory,
)


class SAMConvLSTMCell(BaseConvLSTMCell):
    def __init__(
        self,
        attention_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )

        self.attention_memory = SelfAttentionMemory(out_channels, attention_hidden_dims)

    def forward(
        self,
        X: torch.Tensor,
        prev_h: torch.Tensor,
        prev_cell: torch.Tensor,
        prev_memory: torch.Tensor,
    ) -> Tuple:
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h, new_memory, attention_h = self.attention_memory(new_h, prev_memory)
        return (
            new_h,
            new_cell,
            new_memory,
            attention_h,
        )
