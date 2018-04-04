import os
import functools
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from google.protobuf import text_format

from object_detection.builders import dataset_builder
from object_detection.core import standard_fields as fields
from object_detection.core import prefetcher
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


def get_input_reader_config(tf_record_path):
    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)

    input_reader_proto = input_reader_pb2.InputReader()

    text_format.Merge(input_reader_text_proto, input_reader_proto)

    return input_reader_proto


def normalize_image(image,
                    original_minval,
                    original_maxval,
                    target_minval,
                    target_maxval):
    with tf.name_scope('NormalizeImage', values=[image]):
        original_minval = float(original_minval)
        original_maxval = float(original_maxval)
        target_minval = float(target_minval)
        target_maxval = float(target_maxval)
        factor = (target_maxval - target_minval) / (original_maxval - original_minval)
        image = tf.to_float(image)
        image = tf.subtract(image, original_minval)
        image = tf.multiply(image, factor)
        image = tf.add(image, target_minval)

    return image


def _flip_boxes_left_right(boxes):
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], axis=1)

    return flipped_boxes


def random_adjust_brightness(image, max_delta=0.1):
    with tf.name_scope('RandomAdjustBrightness', values=[image]):
        delta = tf.random_uniform([], -max_delta, max_delta)
        image = tf.image.adjust_brightness(image, delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
    with tf.name_scope('RandomAdjustContrast', values=[image]):
        contrast_factor = tf.random_uniform([], minval=min_delta, maxval=max_delta)
        image = tf.image.adjust_contrast(image, contrast_factor)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_adjust_hue(image, max_delta=0.02):
    with tf.name_scope('RandomAdjustHue', values=[image]):
        delta = tf.random_uniform([], -max_delta, max_delta)
        image = tf.image.adjust_hue(image, delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
    with tf.name_scope('RandomAdjustSaturation', values=[image]):
        saturation_factor = tf.random_uniform([], min_delta, max_delta)
        image = tf.image.adjust_saturation(image, saturation_factor)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_distort_color(image):
    with tf.name_scope('RandomDistortColor', values=[image]):
        image = random_adjust_brightness(image, max_delta=32. / 255.)
        image = random_adjust_saturation(image, min_delta=0.8, max_delta=1.25)
        image = random_adjust_hue(image, max_delta=0.1)
        image = random_adjust_contrast(image, min_delta=0.8, max_delta=1.25)

    return image


def random_crop_to_aspect_ratio(image,
                                boxes,
                                labels,
                                difficult=None,
                                aspect_ratio=21. / 9.,
                                overlap_thresh=0.3):
    with tf.name_scope('RandomCropToAspectRatio', values=[image]):
        image_shape = tf.shape(image)
        orig_height = image_shape[0]
        orig_width = image_shape[1]
        orig_aspect_ratio = tf.to_float(orig_width) / tf.to_float(orig_height)
        target_aspect_ratio = tf.constant(aspect_ratio, dtype=tf.float32)

        def target_height_fn():
            return tf.to_int32(tf.round(tf.to_float(orig_width) / target_aspect_ratio))

        target_height = tf.cond(orig_aspect_ratio >= target_aspect_ratio, lambda: orig_height, target_height_fn)

        def target_width_fn():
            return tf.to_int32(tf.round(tf.to_float(orig_height) * target_aspect_ratio))

        target_width = tf.cond(orig_aspect_ratio <= target_aspect_ratio, lambda: orig_width, target_width_fn)

        offset_height = tf.random_uniform([], minval=0, maxval=orig_height - target_height + 1, dtype=tf.int32)
        offset_width = tf.random_uniform([], minval=0, maxval=orig_width - target_width + 1, dtype=tf.int32)

        new_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

        im_box = tf.stack([
            tf.to_float(offset_height) / tf.to_float(orig_height),
            tf.to_float(offset_width) / tf.to_float(orig_width),
            tf.to_float(offset_height + target_height) / tf.to_float(orig_height),
            tf.to_float(offset_width + target_width) / tf.to_float(orig_width)
        ])

        boxlist = box_list.BoxList(boxes)
        boxlist.add_field('labels', labels)

        if difficult is not None:
            boxlist.add_field('difficult', difficult)

        im_boxlist = box_list.BoxList(tf.expand_dims(im_box, axis=0))

        # remove boxes whose overlap with the image is less than overlap_thresh
        overlapping_boxlist, keep_ids = box_list_ops.prune_non_overlapping_boxes(boxlist, im_boxlist, overlap_thresh)

        # change the coordinate of the remaining boxes
        new_labels = overlapping_boxlist.get_field('labels')
        new_boxlist = box_list_ops.change_coordinate_frame(overlapping_boxlist, im_box)
        new_boxlist = box_list_ops.clip_to_window(new_boxlist, tf.constant([0.0, 0.0, 1.0, 1.0], tf.float32))
        new_boxes = new_boxlist.get()

        result = [new_image, new_boxes, new_labels]

        if difficult is not None:
            new_difficult = new_boxlist.get_field('difficult')
            result.append(new_difficult)

        return tuple(result)


def random_pixel_delta(num, h, w):
    height = tf.to_int64(h)
    width = tf.to_int64(w)
    y = tf.random_uniform(shape=[num], minval=0, maxval=height, dtype=tf.int64)
    x = tf.random_uniform(shape=[num], minval=0, maxval=width, dtype=tf.int64)
    z = tf.zeros(shape=[num], dtype=tf.int64)
    indices = tf.stack([y, x, z], axis=1)

    values = tf.ones([num])
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape=[height, width, 1])

    dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

    return dense_tensor


def random_salt_pepper_noise(image,
                             salt_vs_pepper=0.2,
                             amount=0.0005):
    with tf.name_scope('RandomSaltPepperNoise', values=[image]):
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        size = tf.size(image)

        num_salt = tf.to_int32(tf.ceil(tf.to_float(amount) * tf.to_float(size) * tf.to_float(salt_vs_pepper)))
        num_pepper = tf.to_int32(tf.ceil(tf.to_float(amount) * tf.to_float(size) * tf.to_float(1 - salt_vs_pepper)))

        delta_salt = random_pixel_delta(num_salt, height, width)
        delta_pepper = random_pixel_delta(num_pepper, height, width)
        image = tf.add(image, delta_salt)
        image = tf.subtract(image, delta_pepper)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_horizontal_flip(image, boxes=None):
    result = []

    def _flip_image(image):
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
        do_a_flip_random = tf.random_uniform([])
        do_a_flip_random = tf.greater(do_a_flip_random, 0.5)
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

        result.append(image)

        if boxes is not None:
            boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes), lambda: boxes)
            result.append(boxes)

    return tuple(result)


def preprocess_for_detection(input_dict):
    image = input_dict[fields.InputDataFields.image]
    boxes = input_dict[fields.InputDataFields.groundtruth_boxes]
    labels = input_dict[fields.InputDataFields.groundtruth_classes]

    # normalize
    image = normalize_image(image, 0, 255, 0, 1)

    # random horizontal flip
    image, boxes = random_horizontal_flip(image, boxes)

    # color distort
    image = random_distort_color(image)

    # random crop
    difficult = None
    if fields.InputDataFields.groundtruth_difficult in input_dict:
        difficult = input_dict[fields.InputDataFields.groundtruth_difficult]

    args = [image, boxes, labels, difficult]

    do_a_crop = tf.random_uniform([], minval=0.0, maxval=1.0)
    result = tf.cond(tf.greater(do_a_crop, 0.3),
                     lambda: args,
                     lambda: random_crop_to_aspect_ratio(*args))

    image, boxes, labels = result[:3]
    if difficult is not None:
        difficult = result[3]

    # add noise
    image = random_salt_pepper_noise(image)

    # normalize
    image = normalize_image(image, 0, 1, 0, 255)

    input_dict[fields.InputDataFields.image] = image
    input_dict[fields.InputDataFields.groundtruth_boxes] = boxes
    input_dict[fields.InputDataFields.groundtruth_classes] = labels

    if fields.InputDataFields.groundtruth_difficult in input_dict:
        input_dict[fields.InputDataFields.groundtruth_difficult] = difficult

    return input_dict


if __name__ == '__main__':
    record_path = '/data/demo/demo_train.record'
    preprocess_for_detection(record_path)
