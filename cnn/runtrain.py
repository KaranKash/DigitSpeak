import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
from runeval import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

MAX_EPOCHS = 75.0
# LEARNING_RATE_DECAY_FACTOR = 0.35  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.00001 # Initial learning rate.
# NUM_EPOCHS_PER_DECAY = 7.0 # 350.0 # Epochs after which learning rate decays.

def optimizer(num_batches_per_epoch):
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.00001)
        return increment_step, opt, global_step

# def optimizer(num_batches_per_epoch):
#     with tf.variable_scope("Optimizer"):
#         global_step = tf.Variable(initial_value=0, trainable=False)
#         increment_step = global_step.assign_add(1)
#         decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
#         lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                         global_step,
#                                         decay_steps,
#                                         LEARNING_RATE_DECAY_FACTOR,
#                                         staircase=True)
#         opt = tf.train.MomentumOptimizer(lr,0.9)
#         return increment_step, opt, lr, global_step

def train_network(use_gpu=True, restore_if_possible=True, batch_size=30):
    with tf.device("/cpu:0"):
        # Build graph:
        image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        correct_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
        mismatch_mnist = tf.placeholder(tf.float32, shape=(batch_size, 9, MNIST_DIM))
        # permutation = tf.placeholder(tf.int32, shape=(batch_size))
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, step = optimizer(num_batches_per_epoch)
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            # embeddings, loss, mnist = forward_propagation(image_batch, correct_mnist, mismatch_mnist, permutation, dropout=True, train=True)
            embeddings, loss, mnist, mismatch = forward_propagation(image_batch, correct_mnist, mismatch_mnist, dropout=True, train=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')
        summaries = tf.summary.merge_all()

        # Train:
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog2', sess.graph)  # For logging for TensorBoard

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
                # while ((not coord.should_stop()) and (epoch_count <= MAX_EPOCHS)):
                while not coord.should_stop():
                    labels, spec_activations = sess.run([label_batch, embeddings])
                    labels = labels.flatten().tolist()
                    mnist_batch, mismatch_mnist_batch = generate_mnist_set(labels)
                    # indices = permute_batch(labels)
                    _, batch_loss, i, mnist_set, mismatch_set = sess.run([train, loss, step, mnist, mismatch], feed_dict={
                        correct_mnist: mnist_batch, mismatch_mnist: mismatch_mnist_batch
                    })
                    in_batch = i % num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = num_batches_per_epoch
                    epoch_count = (i // (num_batches_per_epoch+1)) + 1

                    acc = accuracy(labels, spec_activations, mnist_set, mismatch_set)

                    print("Epoch %d. Batch %d/%d. Batch Loss %.2f. Acc %.2f." % (epoch_count, in_batch, num_batches_per_epoch, batch_loss, acc*100))

                    # if i % 50 == 0:
                    #     run_eval(labels, spec_activations, mnist_set)

                    if in_batch == num_batches_per_epoch:
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        print("Saving to %s" % SAVED_MODEL_PATH)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)
                        # evaluate(partition="test")

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

def accuracy(labels, spec_activations, mnist_set, mismatch_set):
    num_correct = 0.
    for i in xrange(len(labels)):
        spec = spec_activations[i].flatten()
        mnist = mnist_set[i].flatten()
        base_score = mnist.dot(spec)
        correct = True
        for mismatch in mismatch_set[i]:
            score = mismatch.flatten().dot(spec)
            if score > base_score:
                correct = False
                break
        if correct:
            num_correct += 1.
    out = num_correct/len(labels)
    return out

def run_eval(labels, spec_activations, mnist_batch):
    correct, total = annotate(labels, spec_activations, mnist_batch)
    print("Annotation accuracy: " + str(100*correct/total) + "%")
    correct, total = retrieve(labels, spec_activations, mnist_batch)
    print("Retrieval accuracy: " + str(100*correct/total) + "%")

if __name__ == "__main__":
    train_network(use_gpu=False)
