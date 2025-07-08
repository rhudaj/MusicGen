import torch
from .rnn.model import MusicRNN
from .rnn.model_batched import MusicRNN_Batched
from util.types import PianoState, NoteSample, PianoStateSamples

'''
NOTE Temperature

The "temperature" parameter scales the logits (the raw, unnormalized output of a model) BEFORE the softmax function is applied.

- **Higher** => makes the output more uniform, increasing the probability of less likely outcomes
- **Lower** => makes the output more peaked, favoring the most likely outcomes.
'''

'''
NOTE `argmax` vs. `multinomial` sampling

Say your note probabilities after softmax are:

```
note_probs = [0.1, 0.6, 0.2, 0.1]
```

- `torch.argmax(note_probs)` Will always return `1` (the highest probability)

- `torch.multinomial(note_probs, 1)` Will return:
	- 1 about 60% of the time
	- 2 about 20% of the time
	- 0 or 3 about 10% of the time each
'''

def sample_note(
	note_logits: torch.Tensor,
 	temperature=0.0
) -> NoteSample:
	'''
	Sample a single note from the note logits

	Args:
		note_logits: raw output of the model
		temperature: controls the randomness of the selected note
		- default=0 => completely deterministic (always choose max probability)
	'''
	note_logits = note_logits.squeeze(0)
	if temperature > 0:
		note_logits = note_logits / temperature

	probs = torch.softmax(note_logits, dim=-1)
	idx = torch.multinomial(probs, 1).item() if temperature > 0 else torch.argmax(probs).item()

	return NoteSample(note=idx, probs=probs.numpy())

def sample_notes(
	model: MusicRNN_Batched,
	start_event: torch.Tensor,
	length=20,
	temperature=0.0,
) -> PianoStateSamples:
	"""
	Generate sequence using classification for notes and regression for durations

	Args:
		start_event: a single note to start the sequence with
		length: length of the output sequence (not including the start event)
		temperature: controls randomness of the selected note
	"""
	model.eval()  # Set to evaluation mode

	output = PianoStateSamples(
		piano_states= [(0, 0)] * length,
		note_samples=[None] * length
	)

	with torch.no_grad():
		hidden = model.init_hidden(batch_size=1)
		current_event = start_event

		for i in range(0, length):

			# STEP 1: Predict next event
			input_event = current_event.unsqueeze(0)  # add batch dimension => Shape: (1, 2)

			note_logits, duration_pred, hidden = model.sample(input_event, hidden)

			# STEP 2: Get duration from regression
			predicted_dur = duration_pred.squeeze(0).item()
			predicted_dur = max(1, int(predicted_dur))  # Ensure positive duration

			# STEP 3: Sample note from classification distribution
			note_sample = sample_note(note_logits, temperature)

			# STEP 4: save event
			piano_state = PianoState((note_sample.note, predicted_dur))
			current_event = torch.tensor(piano_state, dtype=torch.float)

			output.note_samples[i] = note_sample
			output.piano_states[i] = piano_state

	return output