# Piannoroll Dataset

## "LPD-17-Cleansed"

**Cleansed**
- Remove those having more than one time signature change events
- Remove those having a time signature other than 4/4
- Remove those whose first beat not starting from time zero
- Keep only one file that has the highest confidence score in matching for each song

**v17**

The tracks are merged into drums and sixteen instrument families according to the program numbers provided in the MIDI files and the specification of General MIDI (see here). The seventeen tracks are:
- Drums,
- Piano,
- Chromatic Percussion,
- Organ, Guitar,
- Bass,
- Strings,
- Ensemble,
- Brass,
- Reed,
- Pipe,
- Synth Lead,
- Synth Pad,
- Synth Effects,
- Ethnic,
- Percussive
- Sound Effects.

## Structure

The directory structure is a hash-based directory organization system designed to efficiently store and access a large number of files. Here's why it's organized this way:

Purpose of the Structure
File System Performance: Most file systems perform poorly when a single directory contains thousands of files. The nested A-Z structure distributes files across many directories.

Hash Distribution: The nested A-Z directories (3 levels deep) create 26³ = 17,576 possible directory combinations, which helps evenly distribute files.

How It Works
The structure appears to be based on the track ID (like TRAAAGR128F425B14B):

So for track ID TRAAAGR128F425B14B:

1st char: T → but they use A (might be mapped/normalized)
2nd char: R → but they use A
3rd char: A → matches the A
The actual mapping might be:

Taking specific positions from the track ID
Using a hash function on the track ID
Or normalizing certain characters

## Data

### Track Pianoroll

Pianoroll is a music storing format which represents a music piece by a **score-like matrix**.
- **Columns**: note pitch
	- 128 possibilities, covering from C-1 to G9.
- **Rows**: time
	- temporal resolution is set to **24 per beat** in order to cover common temporal patterns such as triplets and 32th notes.
	- Dataset uses **symbolic timing** – the tempo information is removed and thereby each beat has the same length.
- **Values**: velocities

e.g. A bar in 4/4 time with only one track can be represented as a 96 x 128 matrix (96=24*4)

### Multitrack Pianoroll

Set of pianorolls where each pianoroll represents one specific track in the original music piece.

That is, a M-track music piece will be converted into a set of M pianorolls.

e.g., a bar in 4/4 time with M tracks can be represented as a 96 x 128 x M tensor.