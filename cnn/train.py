import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

MAX_EPOCHS = 25.0
LEARNING_RATE_DECAY_FACTOR = 0.35  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00001 # Initial learning rate.
NUM_EPOCHS_PER_DECAY = 7.0 # 350.0 # Epochs after which learning rate decays.

# def optimizer(num_batches_per_epoch):
#     with tf.variable_scope("Optimizer"):
#         global_step = tf.Variable(initial_value=0, trainable=False)
#         increment_step = global_step.assign_add(1)
#         lr = INITIAL_LEARNING_RATE
#         opt = tf.train.AdamOptimizer(lr)
#         return increment_step, opt, lr, global_step

def optimizer(num_batches_per_epoch):
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.MomentumOptimizer(lr,0.9)
        return increment_step, opt, lr, global_step

def train_network(use_gpu=True, restore_if_possible=True, batch_size=128):
    with tf.device("/cpu:0"):
        # Build graph:
        image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        correct_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
        mismatch_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
        permutation = tf.placeholder(tf.int32, shape=(batch_size))
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, lr, step = optimizer(num_batches_per_epoch)
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            embeddings, loss = forward_propagation(image_batch, correct_mnist, mismatch_mnist, permutation, train=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')
        summaries = tf.summary.merge_all()

        # Train:
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

            saver = tf.train.Saver()
            with tf.device("/cpu:0"):
                sess.run(tf.global_variables_initializer())

            if restore_if_possible:
                try:
                    # saver = tf.train.import_meta_graph('./model/model.ckpt-37.meta')
                    saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                    print("Found in-progress model. Will resume from there.")
                except:
                    # saver = tf.train.Saver()
                    # with tf.device("/cpu:0"):
                    #     sess.run(tf.global_variables_initializer())
                    print("Couldn't find old model. Starting from scratch.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            epoch_count = 1
            try:
                while ((not coord.should_stop()) and (epoch_count <= MAX_EPOCHS)):
                    labels, = sess.run([label_batch])
                    labels = labels.flatten().tolist()
                    mnist_batch, mismatch_mnist_batch = generate_mnist_set(labels)
                    indices = permute_batch(labels)
                    _, batch_loss, lrate, i = sess.run([train, loss, lr, step], feed_dict={
                        correct_mnist: mnist_batch, mismatch_mnist: mismatch_mnist_batch, permutation: indices
                    })
                    in_batch = i % num_batches_per_epoch
                    epoch_count = (i // num_batches_per_epoch) + 1

                    print("Epoch %d. Batch %d/%d. LR %.1e. Batch Loss %.2f" % (epoch_count, in_batch, num_batches_per_epoch, lrate, batch_loss))

                    if in_batch + 1 == num_batches_per_epoch:
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        print("Saving to %s" % SAVED_MODEL_PATH)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

if __name__ == "__main__":
    train_network(use_gpu=False)
