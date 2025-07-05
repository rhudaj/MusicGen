# RNN Model – Notes

## Design Choices

1. Separate Heads

- Decoupled Learning: Notes and durations have different characteristics - separate heads allow specialized feature learning for each
- Independent Optimization: Each head can learn optimal representations without interference from the other task
- Scalable Architecture: Easy to add more musical features (velocity, timbre) as additional heads

2. Output Formats

- Notes → Classification: Musical notes are discrete values (0-127), classification naturally handles this without rounding artifacts
- Duration → Regression: Durations can vary continuously and benefit from smooth interpolation between values
- Data-Appropriate: Matches the underlying nature of each musical parameter

3. Loss Functions

- CrossEntropyLoss (Notes): Optimal for multi-class discrete prediction, provides probability distributions for sampling
- MSELoss (Duration): Standard for continuous regression, penalizes large duration errors more heavily
- Combined Loss: Allows joint optimization while respecting each output type's characteristics

## About torch.nn.RNN

parameters:
	input_size (H_in) - number of expected features in the input x
 	hidden_size - number of features in the hidden state h
	num_layers - Number of recurrent layers

# inputs:
	input - Tensor of shape (L, H_in) for unbatches input
	hx - Tensor of shape (D * num_layers * H_out)
		(Defaults to zeros if not provided)

Summary:
When using unbatched inputs with torch.nn.RNN:
- Input shape: (seq_len, input_size)
- Hidden shape: (num_layers, hidden_size) ← No batch dimension
When using batched inputs:
- Input shape: (seq_len, batch_size, input_size)
- Hidden shape: (num_layers, batch_size, hidden_size)