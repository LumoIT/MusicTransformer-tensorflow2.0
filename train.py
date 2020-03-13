from model import MusicTransformerDecoder
import tensorflow as tf
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import argparse
import datetime

from pathlib import Path

import click

@click.command()
@click.option("-l", "--num-layers", default=6, type=int)
@click.option("-s", "--length", default=2048, type=int)

@click.option("-r", "--rate", default=None, type=float)
@click.option("-b", "--batch", default=2, type=int)
@click.option("-e", "--epochs", default=100, type=int)

@click.option("-L", "--load-path", type=click.Path(exists=True, file_okay=False, readable=True))
@click.option("-S", "--save-path", type=click.Path(file_okay=False, writable=True))

@click.argument("preproc-dir", type=click.Path(exists=True, file_okay=False, readable=True))
def train(num_layers, length, rate, batch, epochs, load_path, save_path, preproc_dir):
	if rate is None:
		rate = callback.CustomSchedule(par.embedding_dim)
	preproc_dir = Path(preproc_dir)

	print(rate)
	opt = Adam(rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

	model = MusicTransformerDecoder(
		embedding_dim=256,
		vocab_size=par.vocab_size,
		num_layer=num_layers,
		max_seq=length,
		dropout=0.2,
		debug=False,
		loader_path=load_path,
	)
	model.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)

	# define tensorboard writer
	current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
	eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

	dataset = Data(preproc_dir)

	idx = 0
	with click.progressbar(length=epochs) as prog:
		for e in prog:
			model.reset_metrics()
			for b in range(len(dataset.files) // batch):
				batch_x, batch_y = dataset.slide_seq2seq_batch(batch, length)
				loss, acc = model.train_on_batch(batch_x, batch_y)

				if b % 100 == 0:
					eval_x, eval_y = dataset.slide_seq2seq_batch(batch, length, 'eval')
					(eloss, eacc), weights = model.evaluate(eval_x, eval_y)
					if save_path is not None:
						model.save(save_path)

					with train_summary_writer.as_default():
						if b == 0:
							tf.summary.histogram("target_analysis", batch_y, step=e)
							tf.summary.histogram("source_analysis", batch_x, step=e)

						tf.summary.scalar('loss', loss, step=idx)
						tf.summary.scalar('accuracy', acc, step=idx)

					with eval_summary_writer.as_default():
						if b == 0:
							model.sanity_check(eval_x, eval_y, step=e)

						tf.summary.scalar('loss', eloss, step=idx)
						tf.summary.scalar('accuracy', eacc, step=idx)

						for i, weight in enumerate(weights):
							with tf.name_scope("layer_%d" % i):
								with tf.name_scope("w"):
									utils.attention_image_summary(weight, step=idx)
					idx += 1
					print("\n====================================================")
					print(f"Epoch/Batch: {e}/{b}")
					print(f"Train >>>> Loss: {loss:6.6}, Accuracy: {acc}")
					print(f"Eval >>>> Loss: {eloss:6.6}, Accuracy: {eacc}")


if __name__ == '__main__':
	tf.executing_eagerly()
	train()
