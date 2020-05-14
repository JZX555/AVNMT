import math
import tensorflow as tf

_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1

def _create_min_max_boundaries(
        max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
    """Create min and max boundary lists up to max_length.

    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
        buckets_min = [0, 4, 8, 16, 24]
        buckets_max = [4, 8, 16, 24, 25]

    Args:
        max_length: The maximum length of example in dataset.
        min_boundary: Minimum length in boundary.
        boundary_scale: Amount to scale consecutive boundaries in the list.

    Returns:
        min and max boundary lists
    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = min_boundary

    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    # Create min and max boundary lists from the initial list.
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]

    return buckets_min, buckets_max

def _get_example_length(example):
    """Returns the maximum length between the example inputs and targets."""
    length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])

    return length

def _batch_examples(dataset, batch_size, max_length):
    """Group examples by similar lengths, and return batched dataset.

    Each batch of similar-length examples are padded to the same length, and may
    have different number of elements in each batch, such that:
        group_batch_size * padded_length <= batch_size.

    This decreases the number of padding tokens per batch, which improves the
    training speed.

    Args:
        dataset: Dataset of unbatched examples.
        batch_size: Max number of tokens per batch of examples.
        max_length: Max number of tokens in an example input or target sequence.

    Returns:
        Dataset of batched examples with similar lengths.
    """
    # Get min and max boundary lists for each example. These are used to calculate
    # the `bucket_id`, which is the index at which:
    # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
    # Note that using both min and max lists improves the performance.
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)

    # Create list of batch sizes for each bucket_id, so that
    # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
    bucket_batch_sizes = [batch_size // x for x in buckets_max]

    # bucket_id will be a tensor, so convert this list to a tensor as well.
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
    print('bucket_batch_sizes is: {}'.format(bucket_batch_sizes))

    def example_to_bucket_id(example_input, example_target):
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = _get_example_length((example_input, example_target))

        # TODO(xunkai): investigate if removing code branching improves performance.
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)

        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))