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


## Batch-Training with the `PyTorch` RNN

Batch processing is much more memory and compute efficient because of GPU parallelization across the batch dimension.

### Difference

**Non-Batched:**

- Process 1 sequence at a time -> 1 loss value
- Sum losses across all seq's in training loop (epoch).

**Batched**

- Process multiple sequences simultaneously (in parallel)
- Loss function receives predictions and targets for ALL sequences at once
- PyTorch automatically handles the aggregation based on the `reduction` parameter

With `reduction='sum'`, both approaches give identical results.


### Problem: Variable-Length Sequences

Can use `torch.nn.utils.rnn.pack_sequence` on batches which handles this for you.

- RNN processes the packed sequence efficiently