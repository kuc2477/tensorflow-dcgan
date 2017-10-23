import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data import DATASETS, DATASET_LENGTH_GETTERS
import utils


def _sample_z(sample_size, z_size):
    return np.random\
        .uniform(-1., 1., size=[sample_size, z_size])\
        .astype(np.float32)


def train(model, config, session=None):
    # define session if needed.
    session = session or tf.Session()

    # define summaries.
    summary_writer = tf.summary.FileWriter(config.log_dir, session.graph)
    image_summary = tf.summary.image(
        'generated images', model.G, max_outputs=8
    )
    loss_summaries = tf.summary.merge([
        tf.summary.scalar('discriminator loss', model.d_loss),
        tf.summary.scalar('generator loss', model.g_loss),
    ])

    # define optimizers
    D_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    G_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # get parameter update tasks
    d_grads = D_trainer.compute_gradients(model.d_loss, var_list=model.d_vars)
    g_grads = G_trainer.compute_gradients(model.g_loss, var_list=model.g_vars)
    update_D = D_trainer.apply_gradients(d_grads)
    update_G = G_trainer.apply_gradients(g_grads)

    # main training session context
    with session:
        if config.resume:
            epoch_start = (
                utils.load_checkpoint(session, model, config)
                // DATASET_LENGTH_GETTERS[config.dataset]()
            ) + 1

        else:
            epoch_start = 1
            session.run(tf.global_variables_initializer())

        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(enumerate(dataset, 1))

            for batch_index, xs in dataset_stream:
                # where are we?
                iteration = (epoch-1)*dataset_length + batch_index

                # run the discriminator trainer.
                zs = _sample_z(config.batch_size, model.z_size)
                _, d_loss = session.run(
                    [update_D, model.d_loss],
                    feed_dict={
                        model.z_in: zs,
                        model.image_in: xs
                    }
                )

                # run the generator trainer.
                for _ in range(config.generator_update_ratio):
                    zs = _sample_z(config.batch_size, model.z_size)
                    _, g_loss = session.run(
                        [update_G, model.g_loss],
                        feed_dict={model.z_in: zs}
                    )

                dataset_stream.set_description((
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'g loss: {g_loss:.3f} | '
                    'd loss: {d_loss:.3f}'
                ).format(
                    epoch=epoch,
                    epochs=config.epochs,
                    trained=batch_index*config.batch_size,
                    total=dataset_length,
                    progress=(
                        100.
                        * batch_index
                        * config.batch_size
                        / dataset_length
                    ),
                    g_loss=g_loss,
                    d_loss=d_loss,
                ))

                # log the generated samples.
                if iteration % config.image_log_interval == 0:
                    zs = _sample_z(config.sample_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        image_summary, feed_dict={
                            model.z_in: zs
                        }
                    ), iteration)

                # log the losses.
                if iteration % config.loss_log_interval == 0:
                    zs = _sample_z(config.batch_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        loss_summaries, feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    ), iteration)

                # save the model.
                if iteration % config.checkpoint_interval == 0:
                    utils.save_checkpoint(session, model, iteration, config)
