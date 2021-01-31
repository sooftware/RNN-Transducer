# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor

from rnnt.decoder import DecoderRNNT
from rnnt.encoder import EncoderRNNT


class RNNTransducer(nn.Module):
    """
    RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms.
    Unlike most sequence-to-sequence models, which typically need to process the entire input sequence
    (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and
    streams output symbols, a property that is welcome for speech dictation. In our implementation,
    the output symbols are the characters of the alphabet.

    Args:
        num_classes (int): number of classification
        input_dim (int): dimension of input vector
        num_encoder_layers (int, optional): number of encoder layers (default: 4)
        num_decoder_layers (int, optional): number of decoder layers (default: 1)
        encoder_hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        decoder_hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
        encoder_dropout_p (float, optional): dropout probability of encoder
        decoder_dropout_p (float, optional): dropout probability of decoder
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification

    Inputs: inputs, input_lengths, targets, target_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 1,
            encoder_hidden_state_dim: int = 320,
            decoder_hidden_state_dim: int = 512,
            output_dim: int = 512,
            rnn_type: str = "lstm",
            bidirectional: bool = True,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            sos_id: int = 1,
            eos_id: int = 2,
    ):
        super(RNNTransducer, self).__init__()
        self.num_classes = num_classes
        self.encoder = EncoderRNNT(
            input_dim=input_dim,
            hidden_state_dim=encoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_encoder_layers,
            rnn_type=rnn_type,
            dropout_p=encoder_dropout_p,
            bidirectional=bidirectional,
        )
        self.decoder = DecoderRNNT(
            num_classes=num_classes,
            hidden_state_dim=decoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
            sos_id=sos_id,
            eos_id=eos_id,
            dropout_p=decoder_dropout_p,
        )

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)
        
    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs
