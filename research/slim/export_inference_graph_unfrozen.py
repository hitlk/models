import os
import tensorflow as tf

from nets import nets_factory
from datasets import pascal

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet',
    'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_string(
    'input_type', 'image_tensor',
    'Type of input node. Can be one of [`image_tensor`, `encoded_image_string_tensor`, '
    '`tf_example`]')

tf.app.flags.DEFINE_string(
    'trained_checkpoint_prefix', None,
    'Path to trained checkpoint, typically of the form '
    'path/to/model.ckpt')

tf.app.flags.DEFINE_string(
    'output_directory', None, 'Path to write outputs.')

tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')

FLAGS = tf.app.flags.FLAGS

signature_constants = tf.saved_model.signature_constants


def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(
        dtype=tf.uint8, shape=input_shape, name='image_tensor'
    )

    return input_tensor, input_tensor


def _encoded_image_string_input_placeholder():
    """Returns input that accepts a batch of PNG or JPEG strings."""
    batch_image_str_placeholder = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name='encoded_image_string_tensor'
    )

    def decode(encoded_image_string_tensor):
        image_tensor = tf.image.decode_image(encoded_image_string_tensor, channels=3)
        image_tensor.set_shape((None, None, 3))
        return image_tensor

    return (batch_image_str_placeholder,
            tf.map_fn(
                decode,
                elems=batch_image_str_placeholder,
                parallel_iterations=32,
                back_prop=False
            ))


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor': _encoded_image_string_input_placeholder
}


def _build_graph(input_type, model_name, num_classes, image_size, input_shape):
    """Build the classification graph."""
    if input_type not in input_placeholder_fn_map:
        raise ValueError('Unknown input type: {}'.format(input_type))
    placeholder_args = {}
    if input_shape is not None:
        if input_type != 'image_tensor':
            raise ValueError('Can only specify input shape for `image_tensor` inputs.')
        placeholder_args['input_shape'] = input_shape

    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](**placeholder_args)

    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes,
        is_training=False
    )

    with tf.name_scope('PreProcessing', values=[input_tensors, image_size]):
        image = tf.squeeze(input_tensors, axis=0)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = tf.image.resize_images(image, [image_size, image_size])

        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        images = tf.expand_dims(image, axis=0)

    logits, endpoints = network_fn(images)
    predictions = endpoints['Predictions']
    image_classes = tf.arg_max(predictions, dimension=1)
    image_scores = tf.reduce_max(predictions, axis=1)

    outputs = {
        signature_constants.CLASSIFY_OUTPUT_CLASSES: tf.identity(image_classes, name='image_classes'),
        signature_constants.CLASSIFY_OUTPUT_SCORES: tf.identity(image_scores, name='image_scores')}

    return outputs, placeholder_tensor


def _export_inference_graph(input_type,
                            model_name,
                            num_classes,
                            trained_checkpoint_prefix,
                            output_directory,
                            image_size=None,
                            input_shape=None):
    tf.gfile.MakeDirs(output_directory)
    saved_model_path = os.path.join(output_directory)

    outputs, placeholder_tensor = _build_graph(input_type, model_name, num_classes, image_size, input_shape)

    saver = tf.train.Saver()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        saver.restore(sess, trained_checkpoint_prefix)
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

        tensor_info_inputs = {
            signature_constants.CLASSIFY_INPUTS: tf.saved_model.utils.build_tensor_info(placeholder_tensor)
        }

        tensor_info_outputs = {}
        for k, v in outputs.items():
            tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(outputs[v])

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=signature_constants.CLASSIFY_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            }
        )

        builder.save()


def export_inference_graph(input_type,
                           model_name,
                           dataset_dir,
                           dataset_name,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None):
    dataset = pascal.get_split('train', dataset_name, dataset_dir)
    _export_inference_graph(input_type,
                            model_name,
                            dataset.num_classes,
                            trained_checkpoint_prefix,
                            output_directory, input_shape)


def main(_):
    export_inference_graph(FLAGS.input_type,
                           FLAGS.model_name,
                           FLAGS.dataset_dir,
                           FLAGS.dataset_name,
                           FLAGS.trained_checkpoint_prefix,
                           FLAGS.output_directory)


if __name__ == '__main__':
    tf.app.run()