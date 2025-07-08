import torch as torch
from util.globals import DEVICE

class MusicRNN_Batched(torch.nn.Module):
	def __init__(self,
		num_features=2,
		hidden_size=128,
		num_pitches=129,
		dropout=0,
		num_layers=1,
	):
		super().__init__()

		self.hidden_size = hidden_size
		self.num_pitches = num_pitches

		self.num_layers = num_layers

		# Input: [note_value, duration] - just 2 numbers!
		self.rnn = torch.nn.RNN(
			input_size=num_features,  # note (0-128) + duration (1+)
			hidden_size=hidden_size,
			num_layers=self.num_layers,
			bias=False,
			batch_first=True,
			nonlinearity='tanh',
			dropout=dropout,
		)

		# Separate heads for note classification and duration regression

		# 1. Classification: outputs logits for each note
		self.note_head = torch.nn.Linear(hidden_size, num_pitches)

		# 2. Regression: outputs single duration value
		self.duration_head = torch.nn.Linear(hidden_size, 1)

	def forward(self, packed_input, hidden):
		# packed_input is already a PackedSequence
		packed_output, hidden = self.rnn(packed_input, hidden)

		# Apply output heads directly to packed data
		note_logits = self.note_head(packed_output.data)  # Shape: (total_elements, num_pitches)
		duration_pred = self.duration_head(packed_output.data)  # Shape: (total_elements, 1)

		# Return packed results for easy handling
		packed_note_logits = torch.nn.utils.rnn.PackedSequence(
			note_logits, packed_output.batch_sizes
		)
		packed_duration_pred = torch.nn.utils.rnn.PackedSequence(
			duration_pred, packed_output.batch_sizes
		)

		return packed_note_logits, packed_duration_pred, hidden

	def sample(self, x: torch.Tensor, hidden=None):
		'''
		Sampling mode: expects regular tensor
		x shape: (seq_len, features) or (seq_len, batch_size, features)
  		'''
		if x.dim() == 2:
			x = x.unsqueeze(1)  # Add batch dimension: (seq_len, 1, features)

		rnn_out, hidden = self.rnn(x, hidden)
		note_logits = self.note_head(rnn_out)
		duration_pred = self.duration_head(rnn_out)

		# Remove batch dimension for sampling
		note_logits = note_logits.squeeze(1)  # (seq_len, num_pitches)
		duration_pred = duration_pred.squeeze(1)  # (seq_len, 1)

		return note_logits, duration_pred, hidden

	def init_hidden(self, batch_size: int):
		return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
