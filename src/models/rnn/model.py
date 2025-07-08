import torch as torch
from util.globals import DEVICE

class MusicRNN(torch.nn.Module):
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
			nonlinearity='tanh',
			dropout=dropout,
		)

		# Separate heads for note classification and duration regression

		# 1. Classification: outputs logits for each note
		self.note_head = torch.nn.Linear(hidden_size, num_pitches)

		# 2. Regression: outputs single duration value
		self.duration_head = torch.nn.Linear(hidden_size, 1)

	def forward(self, x, hidden=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# x shape: (seq_len, 2) or (1, 2) for single step
		rnn_out, hidden = self.rnn(x, hidden)

		note_logits = self.note_head(rnn_out)       # Shape: (seq_len, num_pitches)
		duration_pred = self.duration_head(rnn_out) # Shape: (seq_len, 1)

		return note_logits, duration_pred, hidden

	def init_hidden(self):
		return torch.zeros(self.num_layers, self.hidden_size, device=DEVICE)
