# -*- coding: utf-8 -*-
"""
Code to run our optimal model - either using code to translate between assignment labelset and coco labelset
or using finetuning.
"""

import sys
import PIL.Image
from PIL import ImageOps
import numpy as np
import io
import csv
import tensorflow as tf
import os
import glob
import click
import time
from datetime import datetime

MODEL_DIR = "../PretrainedModels/"
RESTORE_DIR = "../Models/"
MODEL_NAME = "faster_rcnn_resnet50_coco_2018_01_28"

LOG_DIR = './log'

def normalize_image(rgb):
  # whiten and center on zero
  white_rgb = rgb - np.mean(rgb)
  white_rgb = white_rgb / np.max(np.abs(white_rgb))
  return white_rgb

def decompress_expand(jpg):
  #zero pad to 320x320
  encoded_jpg_io = io.BytesIO(jpg)
  image = PIL.Image.open(encoded_jpg_io)
  DESIRED_SIZE = 320
  widthpad = (DESIRED_SIZE - image.width)
  heightpad = (DESIRED_SIZE - image.height)
  padding = (widthpad//2, heightpad//2, widthpad - widthpad//2, heightpad-heightpad//2)
  padded = ImageOps.expand(image, padding)
  return padded


def load_and_preprocess(path):
  # load and preprocess 1 file
  with tf.gfile.GFile(path, 'rb') as fp:
    jpg = fp.read()
  return decompress_expand(jpg)

def get_training_image_filenames(split=False):
  # load train images
  train_img_list = glob.glob(r"../Input/train2014/*.jpg")
  if not split:
    return train_img_list
  else:
    np.random.seed(1)
    np.random.shuffle(train_img_list)
    # 90/10 split
    split_pt = (len(train_img_list) // 10) * 9
    return train_img_list[:split_pt], train_img_list[split_pt:]

def get_validation_image_filenames():
  val_img_list = glob.glob(r"../Input/val2014/*.jpg")
  return val_img_list


def MHE(labels, max_label=20):
  #multi hot encode - like one hot encode but can have multiple classes
  output = np.zeros(max_label, dtype=np.float32)
  for lab in labels:
    output[lab] = 1.0 / len(labels)

  return output

#load train labels
train_labels_filename = r"../Input/train.txt"
with open(train_labels_filename) as f:
  reader = csv.reader(f, delimiter='\t')
  train_labels = {k:list(map(int, v.split(','))) for k, v in reader}
  
def get_label(filename):
  # get the correct label for a given filename
  if '/' in filename:
    filename = filename.rsplit('/', 1)[-1]
  return train_labels.get(filename)


class relabeler(object):
  def __init__(self):
    self.labels = {}
    # mapping from cocolabel to tutorlabel + 1 (0 reserved for 'reject bounding box')
    self.labels[9] = 1  # boat
    self.labels[10] = 5  # traffic light
    self.labels[14] = 6  # parking meter
    self.labels[15] = 7  # bench
    self.labels[16] = 18  # bird
    self.labels[28] = 19  # umbrella
    self.labels[31] = 20  # handbag
    self.labels[36] = 2  # snowboard
    self.labels[39] = 4  # baseball bat
    self.labels[41] = 3  # skateboard
    self.labels[44] = 10  # bottle
    self.labels[49] = 9  # knife
    self.labels[50] = 8  # spoon
    self.labels[52] = 15  # banana
    self.labels[54] = 13  # sandwich
    self.labels[57] = 14  # carrot
    self.labels[79] = 17  # oven
    self.labels[80] = 16  # toaster
    self.labels[85] = 12  # clock
    self.labels[90] = 11  # toothbrush
    self.fix_fn = self.get_fix_fn()

  def fix(self, old):
    return self.labels.get(old, 0)

  def get_fix_fn(self):
    return np.vectorize(self.fix)

  def get_best_label(self, preds):
    # convert to new label space and get best label
    labels = self.fix_fn(preds)
    # return labels
    # return first that isn't 0
    if len(np.nonzero(labels)[0]):
      return labels[np.nonzero(labels)[0][0]] - 1  # shift back into original label space
    else:
      return 19  # -1 handbag is the thing we most frequently fail at

def prediction(sess, validation=False, finetuned=False):
  #PREDICTION
  # validation - use validation set
  # else use test part of train/test split
  BATCH_SIZE = 32
  batch_no = 0
  batch_list = []
  result_list = []
  if not validation:
    _, images = get_training_image_filenames(True)
  else:
    images = get_validation_image_filenames()

  start = time.time()
  image_iter = iter(images)
  while image_iter:
    try:
      while len(batch_list) < BATCH_SIZE:
        batch_list.append(load_and_preprocess(next(image_iter)))
    except StopIteration:
      pass
    stacked_imgs = np.stack(batch_list)
    batch_result = sess.run(("detection_classes:0", "detection_scores:0"), feed_dict={"image_tensor:0": stacked_imgs})

    batch_no += 1
    print(".", end="")
    result_list.append(batch_result)

  elapsed = time.time() - start
  results = [x[0] for x in result_list]
  stacked_results = np.vstack(results)
  print("Classified {0} images in {1} seconds".format(len(stacked_results), elapsed))

  if not finetuned:
    #convert preds back into tutor space
    fixer = relabeler()
    preds  = stacked_results
    pred_labels = list(map(fixer.get_best_label, preds))
  else:
    # finetuning model - we have 20 outputs - select one with largest value
    pred_labels = np.argmax(preds, axis=1)

  if not validation:
    #report accuracy
    labelled = zip(images, pred_labels)
    correct = 0
    for fn, pred in labelled:
      truth = train_labels[fn.split("/")[-1]]
      if pred in truth:
        correct += 1
    acc = correct/len(pred_labels)
    print("Test set accuracy: {}".format(acc))
  else:
    #TODO write out txt labels file in their format
    OUTFILE = "../Output/Predicted_labels.txt"
    labelled = zip(images, pred_labels)
    with open(OUTFILE, 'w') as f:
      for filename, pred in labelled:
        if '/' in filename:
          filename = filename.rsplit('/', 1)[-1]
        f.write("{0}\t{1}\n".format(filename, pred))
      print("Wrote {} predictions to output file {}".format(len(pred_labels), OUTFILE))

def fine_tuning(sess):
  # add a single dense relu layer followed by softmax
  label_pl = tf.placeholder(tf.float32, [None, 20], name='labels')
  with tf.name_scope("train"):
    #use reduce_max to find the maximum confidence per class for any bounding box
    reduced_scores = tf.math.reduce_max(sess.graph.get_tensor_by_name("SecondStagePostprocessor/convert_scores:0"), axis=1)
    output = tf.layers.dense(inputs=reduced_scores, units=20, name="output", activation=tf.nn.sigmoid)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label_pl))

  # Get gradients of all trainable variables
  var_list = [v for v in tf.trainable_variables()]
  gradients = tf.gradients(loss, var_list, unconnected_gradients='none')
  gradients = list(zip(gradients, var_list))
  learning_rate = 0.01

  # Create Adam optimizer and train
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

  for gradient, var in gradients:
      tf.summary.histogram(var.name + '/gradient', gradient)
    
  for var in var_list:
      tf.summary.histogram(var.name, var)
    
  tf.summary.scalar('cross_entropy', loss)

  # Evaluation op: Accuracy of the model
  with tf.name_scope("accuracy"):
      correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(label_pl, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Merge all summaries together
  merged_summary = tf.summary.merge_all()

  # Initialize the FileWriter
  writer = tf.summary.FileWriter(LOG_DIR + "/" + str(time.time()))

  # Initialize a saver for storing model checkpoints
  saver = tf.train.Saver()

  BATCH_SIZE = 32
  num_epochs = 10

  batch_no = 0

  checkpoint_path = "../Models/"

  init = tf.global_variables_initializer()
  sess.run(init)

  train, test = get_training_image_filenames(True)
  # Get the number of training/validation steps per epoch
  train_batches_per_epoch = int(np.floor(len(train)/BATCH_SIZE))
  test_batches_per_epoch = int(np.floor(len(test)/ BATCH_SIZE))
  display_step = 50

  def load_batch(from_iter):
    batch_list = []
    labels_list= []
    while len(batch_list) < BATCH_SIZE:
      nextfile = next(from_iter)
      batch_list.append(load_and_preprocess(nextfile))
      labels_list.append(MHE(get_label(nextfile)))
    return np.stack(batch_list), np.stack(labels_list)

  for epoch in range(num_epochs):
    train_iter  = iter(train)
    test_iter = iter(test)
    for step in range(train_batches_per_epoch):
      stacked_imgs, labels = load_batch(train_iter)
      batch_result = sess.run((train_op, loss), feed_dict={"image_tensor:0": stacked_imgs, "labels:0":labels})

      # Generate summary with the current batch of data and write to file
      print(".", end="")
      if step % display_step == 0:
        s = sess.run(merged_summary, feed_dict={"image_tensor:0": stacked_imgs, "labels:0":labels})

        writer.add_summary(s, epoch*train_batches_per_epoch + step)

    print("{} Start validation".format(datetime.now()))

    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch):
        stacked_imgs, labels = load_batch(test_iter)
        acc = sess.run(accuracy, feed_dict={"image_tensor:0": stacked_imgs, "labels:0":labels})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                   test_acc))
    print("{} Saving checkpoint of model...".format(datetime.now()))

    # save checkpoint of the model
    checkpoint_name = os.path.join(checkpoint_path,
                                   'model_epoch'+str(epoch+1)+'.ckpt')
    saver.save(sess, checkpoint_name)

    print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                 checkpoint_name))


@click.command()
@click.option("--finetune/--no-finetune", default=False, help="Add an output layer, and fine-tune to this task")
@click.option("--validation/--no-validation", default=False, help="Classify the validation set (else reports on traint-test split)")
def train_optimal(finetune, validation):
  #load the trainable model from the checkpoint
  sess = tf.Session()

  saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR,MODEL_NAME, 'model.ckpt.meta'))

  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join(MODEL_DIR,MODEL_NAME, 'model.ckpt'))

  # write graph for tensorboard
  tf.summary.FileWriter(LOG_DIR, sess.graph)

  if finetune:
    print("Fine tuning")
    fine_tuning(sess)

  print("Predicting on {} set".format("validation" if validation else "test"))
  prediction(sess, validation, finetune)

# Just use dev.env variables
if __name__ == '__main__':
    train_optimal(sys.argv[1:])