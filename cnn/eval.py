import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

def evaluate(partition="train", batch_size=100):
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(1)
        with tf.device("/cpu:0"):
            # Build graph:
            image_batch, label_batch, num_examples_per_epoch = input_graph(training=False, partition=partition, batch_size=batch_size)
            correct_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
            mismatch_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
            permutation = tf.placeholder(tf.int32, shape=(batch_size))
            img = tf.summary.image("image", image_batch, 1)
            with tf.device("/cpu:0"): # Potentially gpu
                embeddings, _ = forward_propagation(image_batch, label_batch, correct_mnist, mismatch_mnist, permutation)

            restorer = tf.train.Saver()  # For saving the model
            acc_annotation, acc_retrieval = 0, 0

            with tf.Session(config=tf.ConfigProto(
                    log_device_placement=False)) as sess:
                # Initialize the variables (like the epoch counter).
                with tf.device("/cpu:0"): # Initialize variables on the main cpu
                    sess.run(tf.global_variables_initializer())

                restorer.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                # Start input enqueue threads.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    # image annotation task
                    i = 0
                    num_correct = 0
                    while ((i * batch_size < num_examples_per_epoch) and not coord.should_stop()):
                        labels, spec_activations = sess.run([label_batch, embeddings])
                        print(spec_activations)
                    #     classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
                    #     for cls, prb in zip(classes, pr[0,:].tolist()):
                    #         print "%s: %6.2f%%" % (cls, prb * 100.0)
                    #     num_correct += current_correct
                        i += 1
                    #
                    # total = i * batch_size
                    # acc_annotation = num_correct / float(total)

                    # image retrieval task
                    #TODO

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                sess.close()

            return acc_annotation, acc_retrieval

def main(batch_size=100):
    acc_annotation, acc_retrieval = evaluate(partition="train")
    print("Image annotation accuracy: %.3f" % (100.0 * acc_annotation))
    print("Image retrieval accuracy: %.3f" % (100.0 * acc_retrieval))

if __name__ == "__main__":
    main()
