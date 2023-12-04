import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn

sys.path.append(".")
from core.utils.constants import DEVICE
from core.attention.SelfAttentionConvLSTM.SAConvLSTM import SAConvLSTM  # noqa: E402

class SASeq2Seq(nn.Module):
    """The sequence to sequence model implementation using Base Self-Attention ConvLSTM."""

    def __init__(
        self,
        attention_hidden_dims: int,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        input_seq_length: int,
        out_channels: Optional[int] = None,
        return_sequences: bool = False,
    ) -> None:
        """

        Args:
            num_channels (int): [Number of input channels]
            kernel_size (int): [kernel size]
            num_kernels (int): [Number of kernels]
            padding (Union[str, Tuple]): ['same', 'valid' or (int, int)]
            activation (str): [the name of activation function]
            frame_size (Tuple): [height and width]
            num_layers (int): [the number of layers]
        """
        super(SASeq2Seq, self).__init__()
        self.attention_hidden_dims = attention_hidden_dims
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.input_seq_length = input_seq_length
        self.out_channels = out_channels if out_channels is not None else num_channels
        self.return_sequences = return_sequences

        self.sequential = nn.Sequential()
        #output_width = (input_width - kernel_width + 2 * padding) / stride + 1
        #self.convfirst=nn.Conv2d(num_channels,num_kernels,kernel_size=5,stride=2,padding=1)

        # Add first layer (Different in_channels than the rest)
        self.sequential.add_module(
            "sa_convlstm1",
            SAConvLSTM(
                attention_hidden_dims=self.attention_hidden_dims,
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
            ),
        )

        self.sequential.add_module(
            "layernorm1",
            nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
        )

        # Add the rest of the layers
        for layer_idx in range(2, num_layers + 1):
            self.sequential.add_module(
                f"sa_convlstm{layer_idx}",
                SAConvLSTM(
                    attention_hidden_dims=self.attention_hidden_dims,
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                ),
            )

            self.sequential.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
            )

        self.sequential.add_module(
            "conv3d",
            nn.Conv3d(
                in_channels=self.num_kernels,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                padding="same",
            ),
        )

        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        #past_seq_len=self.config.input_seq_len #输入的帧
        #output = self.sequential(X[:,:,:past_seq_len])#b,c,s,h,w
        output = self.sequential(X)
        if self.return_sequences is True:
            return output

        return output[:, :, -1:, ...]

    def get_attention_maps(self):
        # get all sa_convlstm module
        sa_convlstm_modules = [
            (name, module)
            for name, module in self.named_modules()
            if module.__class__.__name__ == "SAConvLSTM"
        ]
        return {
            name: module.attention_scores for name, module in sa_convlstm_modules
        }  # attention scores shape is (batch_size, seq_length, height * width)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    input_X = torch.rand((5, 3, 6, 16, 16), dtype=torch.float, device=DEVICE)
    model = (
        SASeq2Seq(
            attention_hidden_dims=4,
            num_channels=3,
            kernel_size=3,
            num_kernels=4,
            padding="same",
            activation="relu",
            frame_size=(16, 16),
            num_layers=4,
            input_seq_length=6,
            return_sequences=True,
        )
        
        .to(torch.float)
    )
    print(model)
    y = model.forward(input_X)
    print(y.shape)
