import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_%s.record'
_NUM_SAMPLES_PATTERN = 'NUM_%s_SAMPLES'


def get_split(split_name, dataset_name, dataset_dir):
    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % (dataset_name, split_name))
    reader = tf.TFRecordReader

    keys_to_feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_feature, items_to_handlers
    )

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    num_classes = len(labels_to_names.keys())
    num_samples = os.environ.get(split_name.upper(), 0)
    num_samples = int(num_samples)
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions={},
        labels_to_names=labels_to_names
    )


if __name__ == '__main__':
    dataset = get_split('train', 'login', '/data/login')
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              common_queue_capacity=20,
                                                              common_queue_min=10)
    [image, label] = provider.get(['image', 'label'])
    with tf.train.MonitoredSession() as sess:
        for i in range(300):
            result = sess.run(label)
            print('%d : %d' % (i, result))