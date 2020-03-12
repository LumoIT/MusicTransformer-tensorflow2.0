import pickle
import click
from midi_processor.processor import encode_midi
from pathlib import Path

@click.command()
@click.argument("midi-dir",    type=click.Path(exists=True, file_okay=False, readable=True), callback=lambda c,p,v: Path(v))
@click.argument("preproc-dir", type=click.Path(             file_okay=False, writable=True), callback=lambda c,p,v: Path(v))
def preprocess(midi_dir, preproc_dir):
	preproc_dir.mkdir(exist_ok=True, parents=True)
	if list(preproc_dir.iterdir()):
		print(f"'{preproc_dir}' is not empty. Continuing anyway.")

	files = []
	files.extend(midi_dir.glob("*.mid"))
	files.extend(midi_dir.glob("*.midi"))
	files.sort()

	# TODO: make it output only one file, in tf format
	with click.progressbar(files, label="Processing", item_show_func=str) as prog:
		for file in prog:
			outfile = preproc_dir / (file.name + ".pickle")
			with file.open("rb") as f:
				data = encode_midi(f)
			with outfile.open("wb") as f:
				pickle.dump(data, f)

if __name__ == '__main__':
	preprocess()
