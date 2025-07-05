import os, glob
import pypianoroll as pr
from typing import Optional
from ..util.globals import num_pitches
from ..util.convert import convert_pianoroll_to_piano_states

data_dir = 'lpd_17_cleansed'

def get_all_npz_files():
	"""Get all .npz files from the nested directory structure"""
	pattern = os.path.join(data_dir, "*", "*", "*", "*", "*.npz")
	return glob.glob(pattern)

def load_multi_track(file_path) -> pr.Multitrack:
	return pr.load(f'{data_dir}/{file_path}')

def get_track_by_instrument(
	multi_track: pr.Multitrack,
	instrument='Bass'
) -> Optional[pr.Track]:

	found_tracks = list(filter(
		lambda t: t.name == instrument,
		multi_track.tracks
	))

	if not found_tracks or len(found_tracks) == 0:
		return None

	return found_tracks[0]