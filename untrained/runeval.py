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
        with tf.device("/cpu:0"):
            # Build graph:
            image_batch, label_batch, num_examples_per_epoch = input_graph(training=False, partition=partition, batch_size=batch_size)
            correct_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
            mismatch_mnist = tf.placeholder(tf.float32, shape=(batch_size, MNIST_DIM))
            # permutation = tf.placeholder(tf.int32, shape=(batch_size))
            img = tf.summary.image("image", image_batch, 1)
            with tf.device("/cpu:0"): # Potentially gpu
                embeddings, _, mnist = forward_propagation(image_batch, correct_mnist, mismatch_mnist, train=False)

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
                    counter = 0
                    while ((counter * batch_size < num_examples_per_epoch) and not coord.should_stop()):
                        labels, spec_activations, mnist_set = sess.run([label_batch, embeddings, mnist])
                        labels = labels.flatten().tolist()
                        for i in range(len(labels)):
                            mnist = mnist_set[i]
                            max_score, max_index = None, None
                            for j in range(len(labels)):
                                score = mnist.dot(spec_activations[j])
                                if score > max_score:
                                    max_score = score
                                    max_index = j
                            match = (get_label_map(labels[i]) == get_label_map(labels[max_index]))
                            if match:
                                num_correct += 1
                            else:
                                print("Mistmatch: MNIST " + str(get_label_map(labels[i])) + " returned " + labels[max_index] + " index " + str(max_index))
                                pass
                        counter += 1

                    total = counter * batch_size
                    acc_annotation = num_correct / float(total)

                    # # image retrieval task
                    # counter = 0
                    # while ((counter * batch_size < num_examples_per_epoch) and not coord.should_stop()):
                    #     labels, spec_activations = sess.run([label_batch, embeddings])
                    #     labels = labels.flatten().tolist()
                    #     for i in range(len(labels)):
                    #         label = labels[i]
                    #         spec_act = spec_activations[i]
                    #         max_score, mnist_label = None, None
                    #         for j in range(10):
                    #             for k in range(10):
                    #                 mnist = get_mnist_embedding((0,j))[1]
                    #                 score = mnist.dot(spec_act)
                    #                 # print(j, score, label)
                    #                 if score > max_score:
                    #                     max_score = score
                    #                     mnist_label = j
                    #         match = (get_label_map(labels[i]) == mnist_label)
                    #         if match:
                    #             num_correct += 1
                    #         else:
                    #             print("Mistmatch: spec " + labels[i] + " returned " + str(mnist_label))
                    #             pass
                    #     counter += 1
                    #
                    # total = counter * batch_size
                    # acc_retrieval = num_correct / float(total)

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                sess.close()

            return acc_annotation, acc_retrieval

def annotate(labels, spec_activations, mnist_batch):
    num_correct = 0
    for i in range(len(labels)):
        mnist = mnist_batch[i]
        max_score, max_index = None, None
        for j in range(len(labels)):
            score = mnist.dot(spec_activations[j])
            if score > max_score:
                max_score = score
                max_index = j
        match = (get_label_map(labels[i]) == get_label_map(labels[max_index]))
        if match:
            num_correct += 1
    return num_correct, len(labels)

def retrieve(labels, spec_activations, mnist_batch):
    num_correct = 0
    for i in range(len(labels)):
        spec = spec_activations[i].flatten()
        max_score, max_index = None, None
        for j in range(len(mnist_batch)):
            mnist = mnist_batch[j].flatten()
            score = mnist.dot(spec)
            if score > max_score:
                max_score = score
                max_index = j
        match = (get_label_map(labels[i]) == get_label_map(labels[max_index]))
        if match:
            num_correct += 1
    return num_correct, len(labels)

def main(batch_size=100):
    acc_annotation, acc_retrieval = evaluate(partition="test")
    print("Image annotation accuracy: %.3f" % (100.0 * acc_annotation))
    print("Image retrieval accuracy: %.3f" % (100.0 * acc_retrieval))

if __name__ == "__main__":
    main()
