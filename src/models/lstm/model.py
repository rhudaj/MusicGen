import torch as torch
from torch.nn.utils.rnn import PackedSequence
from util.globals import DEVICE

class MusicLSTM(torch.nn.Module):
	def __init__(self,
		num_features=2,
		hidden_size=128,
		num_pitches=129,
		dropout=0,
		num_layers=1,
		bidirectional=False,
	):
		super().__init__()

		self.hidden_size = hidden_size
		self.num_pitches = num_pitches

		self.num_layers = num_layers

		self.D = 2 if bidirectional else 1

		# Input: [note_value, duration] - just 2 numbers!
		self.rnn = torch.nn.LSTM(
			input_size=num_features,  # note (0-128) + duration (1+)
			hidden_size=hidden_size,
			num_layers=self.num_layers,
			bias=False,
			batch_first=True,
			dropout=dropout,
			bidirectional=bidirectional
		)

		# SEPERATE HEADS:

		# 1. Note Classification: outputs logits for each note
		self.note_head = torch.nn.Linear(self.D * hidden_size, num_pitches)

		# 2. DurationRegression: outputs single duration value
		self.duration_head = torch.nn.Linear(self.D * hidden_size, 1)

	def forward(self, packed_input: PackedSequence, input_states: tuple):

		output_packed, output_states = self.rnn(packed_input, input_states)

		# `output_packed.data`` shape: (total_elements, D * hidden_size)
  		# NOTE: `total_elements` = sum of all sequence lengths in the batch

		# Apply output heads directly to packed data
		note_logits = self.note_head(output_packed.data)  		# Shape: (total_elements, num_pitches)
		duration_pred = self.duration_head(output_packed.data)  # Shape: (total_elements, 1)

		# Return packed results for easy handling
		packed_note_logits = PackedSequence(
			note_logits, output_packed.batch_sizes
		)

		packed_duration_pred = PackedSequence(
			duration_pred, output_packed.batch_sizes
		)

		return packed_note_logits, packed_duration_pred, output_states

	def sample(self, x: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor]=None):
		'''
		Sampling mode: expects regular tensor
		x shape: (seq_len, features) or (seq_len, batch_size, features)
  		'''
		if x.dim() == 2:
			x = x.unsqueeze(1)  # Add batch dimension: (seq_len, 1, features)

		rnn_out, hidden_state = self.rnn(x, hidden_state)
		note_logits = self.note_head(rnn_out)
		duration_pred = self.duration_head(rnn_out)

		# Remove batch dimension for sampling
		note_logits = note_logits.squeeze(1)  # (seq_len, num_pitches)
		duration_pred = duration_pred.squeeze(1)  # (seq_len, 1)

		return note_logits, duration_pred, hidden_state

	def init_hidden(self, batch_size: int = 1):
		'''
		Creates the initial hidden state & cell state.

		Returns: tuple of Tensor's (hidden, cell state)
  		'''
		size = (self.D * self.num_layers, batch_size, self.hidden_size)
		return torch.zeros(*size, device=DEVICE),  torch.zeros(*size, device=DEVICE)