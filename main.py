#!/usr/bin/env python3
import os.path
import pprint
import tensorflow as tf
from data import DATASETS
from model import DCGAN
from train import train
import utils


flags = tf.app.flags
flags.DEFINE_string('dataset', 'mnist', 'dataset to use {}'.format(
    DATASETS.keys()
))
flags.DEFINE_bool('resize', True, 'whether to resize images on the fly or not')
flags.DEFINE_bool('crop', True,
                  'whether to use crop for image resizing or not')

flags.DEFINE_integer('z_size', 100, 'size of latent code z [100]')
flags.DEFINE_integer(
    'g_filter_number', 64,
    'number of generator\'s filters at the last transposed conv layer'
)
flags.DEFINE_integer(
    'd_filter_number', 64,
    'number of discriminator\'s filters at the first conv layer'
)
flags.DEFINE_integer('g_filter_size', 5, 'generator\'s filter size')
flags.DEFINE_integer('d_filter_size', 4, 'discriminator\'s filter size')

flags.DEFINE_float('learning_rate', 2e-05, 'learning rate for Adam [2e-05]')
flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam [0.5]')
flags.DEFINE_integer('epochs', 10, 'epochs to train')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('sample_size', 36, 'generator sample size')

flags.DEFINE_integer(
    'loss_log_interval', 30,
    'number of batches per logging losses'
)
flags.DEFINE_integer(
    'image_log_interval', 30,
    'number of batches per logging sample images'
)
flags.DEFINE_integer(
    'checkpoint_interval', 1000, 'number of batches per saving the model'
)
flags.DEFINE_integer(
    'generator_update_ratio', 2,
    'number of updates for generator parameters per discriminator updates'
)

flags.DEFINE_bool('test', False, 'flag defining whether it is in test mode')
flags.DEFINE_bool('resume', False, 'whether to resume training or not')
flags.DEFINE_string('log_dir', 'logs', 'directory of summary logs')
flags.DEFINE_string('sample_dir', 'figures', 'directory of generated figures')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'directory of trained models')
FLAGS = flags.FLAGS


def _patch_flags_with_dataset(flags_):
    flags_.image_size = (
        DATASETS[flags_.dataset].image_size or
        flags_.image_size
    )
    flags_.channel_size = (
        DATASETS[flags_.dataset].channel_size or
        flags_.channel_size
    )
    return flags_


def main(_):
    global FLAGS

    # patch and display flags with dataset's width and height
    FLAGS = _patch_flags_with_dataset(FLAGS)
    pprint.PrettyPrinter().pprint(FLAGS.__flags)

    # compile the model
    dcgan = DCGAN(
        label=FLAGS.dataset,
        z_size=FLAGS.z_size,
        image_size=FLAGS.image_size,
        channel_size=FLAGS.channel_size,
        g_filter_number=FLAGS.g_filter_number,
        d_filter_number=FLAGS.d_filter_number,
        g_filter_size=FLAGS.g_filter_size,
        d_filter_size=FLAGS.d_filter_size,
    )

    # test / train the model
    if FLAGS.test:
        with tf.Session() as sess:
            name = '{}_test_figures'.format(dcgan.name)
            utils.load_checkpoint(sess, dcgan, FLAGS)
            utils.test_samples(sess, dcgan, name, FLAGS)
            print('=> generated test figures for {} at {}'.format(
                dcgan.name, os.path.join(FLAGS.sample_dir, name)
            ))
    else:
        train(dcgan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
