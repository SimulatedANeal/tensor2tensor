from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np
import os
import random
import threading
import zipfile
from collections import namedtuple, Counter
from math import ceil
from six.moves import xrange

# Dependency imports
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
OOV = "<UNK>"

URL = "http://codh.rois.ac.jp/char-shape/book/"

_TRAIN_BOOKS = [
    # "bib_id",  # num_page_images, types, tokens
    "200003076", # 346, 1720, 63959
    "200003967", #  88, 1119, 11197
    "200014740", # 182, 1969, 44832
    "200021637", #  37,  417,  4871
    "200021660", # 185, 1758, 32525
    "200021712", # 165,  843, 24480
    "200021763", # 100,  704, 11397
    "200021802", # 111,  560, 19575
    "200021851", #  59,  430,  5599
    "200021853", #  79,  595,  9046
    "200021869", #  35,  330,  3003
    "200021925", #  45,  693,  4259
    "200022050", #  30,  255,  9545
    "brsk00000"  # 238, 2197, 75462
]
_EVAL_BOOKS = [

]
_TEST_BOOKS = [
    "hnsd00000" # 522, 1972, 83492
]
assert all([b not in _TRAIN_BOOKS for b in _EVAL_BOOKS + _TEST_BOOKS])
_ALL_BOOKS = _TRAIN_BOOKS + _EVAL_BOOKS + _TEST_BOOKS

MAX_CONCURRENT_THREADS = 8

Image = namedtuple('Image', ['filename', 'x', 'y', 'w', 'h', 'label'])

def _maybe_get_pmjtc_dataset(directory):
    for bib_id in _ALL_BOOKS:
        if not tf.gfile.Exists(os.path.join(directory, bib_id)):
            filename = bib_id + '.zip'
            download_url = os.path.join(URL, bib_id, filename)
            path = generator_utils.maybe_download(directory, filename,
                                                  download_url)
            unzip_dir = os.path.join(directory, filename.strip(".zip"))
            if not tf.gfile.Exists(unzip_dir):
                zipfile.ZipFile(path, "r").extractall(directory)

def _split_into_shards(meta_list, num_shards):
    shards = []
    spacing = np.linspace(0, len(meta_list), num_shards + 1).astype(np.int)
    for i in xrange(len(spacing) - 1):
        shards.append(meta_list[spacing[i]:spacing[i+1]])
    return shards

def _example_generator(meta_list, shape, encoder, helper):
    """

    Args:
        meta_list:
        shape:
        encoder:
        helper:

    Returns:

    """
    for image in meta_list:
        label = encoder.encode(" ".join(image.label))
        image = _load_image(image, shape, helper)
        height, width = image.shape[:2]
        encoded_image_data = image.tostring()
        yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["raw"],
            "image/class/label": label,
            "image/height": [height],
            "image/width": [width]
        }

def _load_image(img, shape, image_helper):
    """Loads and resizes full or bbox-specified image.

    Args:
        img: Image namedtuple.
        shape: Shape to which image (model input) is resized.
        image_helper: ImageTFOps object

    Returns:
        RGB image.
    """

    with tf.gfile.FastGFile(img.filename, 'rb') as f:
        encoded_image = f.read()

    try:
        image = image_helper.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % img.filename)
        return
    x, y, w, h = img.x, img.y, img.w, img.h
    if x: image = image[y:y + h, x:x + w, :]
    image = _resize(image, shape, image_helper)

    return image

def _resize(image, shape, image_helper):
    """ Resizes images, scales if one dimension None.

    Args:
        image: Image to be resized.
        shape: A list for output dimensions, one unspecified allowed.
            [height, width]
        image_helper: ImageTFOps object

    Returns:
        A resized image

    """
    assert len(shape) == 2
    assert not (shape[0] == None and shape[1] == None), (
        "At least one shape dimension must be provided."
    )

    if shape[0] == None:
        # fixed width, variable height
        w = image.shape[1]
        scale = float(shape[1]) / w
        new_shape = [image.shape[0] * scale, w * scale]
    elif shape[1] == None:
        # fixed height, variable width
        h = image.shape[0]
        scale = float(shape[0]) / h
        new_shape = [h * scale, image.shape[1] * scale]
    else:
        # both fixed
        new_shape = [shape[0], shape[1]]
    new_img = image_helper.resize(image, new_shape)
    return new_img

class ImageTFOps(object):
    """Helper class for decoding and resizing images."""

    def __init__(self):
        # Create a single TensorFlow Session for all image op calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg,
                                                 channels=3)

        # TensorFlow ops for image resizing
        self._image_orig = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3))
        self._shape = tf.placeholder(dtype=tf.int32, shape=(2,))
        self._image_resize = tf.cast(tf.image.resize_images(
            self._image_orig, size=self._shape), tf.uint8)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def resize(self, image, shape):
        image = self._sess.run(
            self._image_resize,
            feed_dict={
                self._image_orig: image,
                self._shape: shape
            }
        )
        return image

class OcrPmjtProblem(image_utils.Image2TextProblem):
    """OCR on Center for Open Data in the Humanities
    Pre-modern Japanese Text Character Shapes dataset"""

    image_helper = ImageTFOps()

    def get_helper(self):
        return self.image_helper

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        _maybe_get_pmjtc_dataset(tmp_dir)
        self._maybe_save_image_meta(data_dir, tmp_dir)
        self._maybe_build_vocab(data_dir, tmp_dir)
        train_meta = self._load_image_meta(data_dir, 'train')
        dev_meta = self._load_image_meta(data_dir, 'dev')
        encoder = text_encoder.TokenTextEncoder(
            os.path.join(data_dir, self.vocab_name),
            replace_oov=OOV
        )

        train_paths = self.training_filepaths(
            data_dir, self.train_shards, shuffled=False)
        dev_paths = self.dev_filepaths(
            data_dir, self.dev_shards, shuffled=False)
        train_meta_shards = _split_into_shards(train_meta, self.train_shards)
        dev_meta_shards = _split_into_shards(dev_meta, self.dev_shards)

        datasets = ((train_meta_shards, train_paths),
                    (dev_meta_shards, dev_paths))

        all_paths = []
        threads = []
        thread_counter = 0
        for i in xrange(len(datasets)):
            for j in xrange(len(datasets[i][0])):
                meta_list = datasets[i][0][j]
                out_file = datasets[i][1][j]
                all_paths.append(out_file)
                t = threading.Thread(
                    target=self.generate_data_shard,
                    args=(thread_counter, meta_list, out_file, encoder)
                )
                threads.append(t)
                thread_counter += 1

        num_batches = int(
            ceil(float(len(threads)) / MAX_CONCURRENT_THREADS))
        for i in xrange(num_batches):
            coord = tf.train.Coordinator()
            start = i * MAX_CONCURRENT_THREADS
            end = start + MAX_CONCURRENT_THREADS
            current = threads[start:end]
            for t in current:
                t.start()
            coord.join(current)

        generator_utils.shuffle_dataset(all_paths)

    def generate_data_shard(self, thread_ix, meta_list, out_file, encoder):
        tf.logging.info("[thread %d], %d image-label pairs" %
                        (thread_ix, len(meta_list)))

        generator_utils.generate_files(
            _example_generator(meta_list, self.image_shape,
                               encoder, self.get_helper()),
            [out_file]
        )

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.input_modality = {"inputs": (registry.Modalities.IMAGE, 256)}
        encoder = self._encoders["targets"]
        p.target_modality = (registry.Modalities.SYMBOL, encoder.vocab_size)
        p.batch_size_multiplier = 256
        p.max_expected_batch_size_per_shard = 4
        p.loss_multiplier = 1.0
        p.input_space_id = problem.SpaceID.IMAGE
        p.target_space_id = self.target_space_id
        p.batch_size = 10

    def dataset_filename(self):
        return self.name

    def feature_encoders(self, data_dir):
        if self.is_character_level:
            encoder = text_encoder.ByteTextEncoder()
        else:
            vocab_filename = os.path.join(data_dir, self.vocab_name)
            encoder = text_encoder.TokenTextEncoder(vocab_filename)
        input_encoder = text_encoder.ImageEncoder()
        return {"inputs": input_encoder, "targets": encoder}

    def example_reading_spec(self):
        data_fields = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/format": tf.FixedLenFeature([], tf.string),
            # "image/class/label": tf.FixedLenSequenceFeature([], tf.int64),
            "image/class/label": tf.VarLenFeature(tf.int64),
        }

        data_items_to_decoders = {
            "inputs":
                tf.contrib.slim.tfexample_decoder.Image(
                    image_key="image/encoded",
                    format_key="image/format",
                    channels=self.num_channels),
            "targets":
                tf.contrib.slim.tfexample_decoder.Tensor("image/class/label")
        }

        return data_fields, data_items_to_decoders

    def preprocess_example(self, example, mode, hparams):
        """Runtime preprocessing.

        Return a dict or a tf.Data.Datset.from_tensor_slices
        (if you want each example to turn into multiple).

        Args:
          example: dict, features
          mode: tf.estimator.ModeKeys
          hparams: HParams, model hyperparameters

        Returns:
          dict or Dataset
        """
        img = example["inputs"]
        img = tf.to_int64(tf.image.rgb_to_grayscale(img))
        img = tf.image.per_image_standardization(img)
        example["inputs"] = img
        return example

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.EDIT_DISTANCE
        ]

    @property
    def batch_size_means_tokens(self):
        return False

    @property
    def is_character_level(self):
        return False

    @property
    def image_scope(self):
        """OCR input image scope.

        Returns a string ID for the problem's image scope,
        e.g. 'single_char', 'full_image', 'sequence'.

        Must be overwritten by subclass problem.

        Returns:
            a string
        """
        raise NotImplementedError()

    @property
    def image_shape(self):
        """Input image shape

        Must be overwritten by subclass.
        One element allowed to be None.

        Returns:
            A tuple of length 2 (height, width).

        """
        raise NotImplementedError

    @property
    def num_targets(self):
        """Number of characters/tokens in input image.

        Defaults to 3 based on PRMU Alcon 2017 [1], task level 2.
        Only used if self.image_scope == 'sequence'.

        Returns:
            an integer

        References:
            [1] https://sites.google.com/view/alcon2017prmu
        """
        return 3

    @property
    def overlapping(self):
        """Overlapping sequences.

        Allow input image crops to overlap.
        Only used if self.image_scope == 'sequence'.

        Returns:
            a boolean

        """
        return True

    @property
    def meta_filename(self):
        """Image metadata filename

        Returns:
            a string

        """
        fname = self.image_scope
        if fname == 'sequence':
            fname += '_%d' % self.num_targets
            if self.overlapping:
                fname += '_overlapping'
        fname += '.json'
        return fname

    @property
    def vocab_name(self):
        return "vocab.pmjt.%d" % self.vocab_size

    @property
    def vocab_size(self):
        return 3999

    @property
    def target_space_id(self):
        return problem.SpaceID.JP_TOK

    @property
    def train_shards(self):
        return 100

    @property
    def dev_shards(self):
        return 20

    def _maybe_save_image_meta(self, data_dir, tmp_dir):
        """Save image_meta dictionary as json with the following format,

        {'train': list of Image namedtuples._asdict() for training set,
         'dev': list of Image namedtuples._asdict() for dev set}

        Args:
            bib_ids:
            data_dir:
            tmp_dir:

        """
        if not tf.gfile.Exists(os.path.join(data_dir, self.meta_filename)):
            meta_dict = {}
            train_meta = self._create_image_meta(_TRAIN_BOOKS, tmp_dir)
            if len(_EVAL_BOOKS) > 0:
                dev_meta = self._create_image_meta(_EVAL_BOOKS, tmp_dir)
            else:
                dev_split = int(len(train_meta) * self.train_shards / (
                        self.train_shards + self.dev_shards))
                random.shuffle(train_meta)
                dev_meta = train_meta[dev_split:]
                train_meta = train_meta[:dev_split]
            meta_dict['train'] = [i._asdict() for i in train_meta]
            meta_dict['dev'] = [i._asdict() for i in dev_meta]
            meta_dict['test'] = [] # TODO
            with open(os.path.join(data_dir, self.meta_filename), 'w') as fp:
                json.dump(meta_dict, fp)

    def _maybe_build_vocab(self, data_dir, allow_singletons=True):
        vocab_file = os.path.join(data_dir, self.vocab_name)
        if not tf.gfile.Exists(vocab_file):
            tf.logging.info("Building vocabulary...")
            token_counts = Counter()
            train_meta = self._load_image_meta(data_dir, 'train')
            for image in train_meta:
                for l in image.label:
                    token_counts[l] += 1
            tokens, counts = zip(*token_counts.most_common(self.vocab_size))
            tokens = (OOV,) + tokens
            vocab_encoder = text_encoder.TokenTextEncoder(None,
                                                          vocab_list=tokens,
                                                          num_reserved_ids=3)
            vocab_encoder.store_to_file(vocab_file)
            tf.logging.info("Saved in %s" % vocab_file)

    def _load_image_meta(self, data_dir, subset):
        with open(os.path.join(data_dir, self.meta_filename), 'r') as fp:
            meta_dict = json.load(fp)
        result = [Image(**i) for i in meta_dict[subset]]
        return result

    def _create_image_meta(self, bib_ids, tmp_dir):
        image_meta = []
        for bib in bib_ids:
            annotation_file = os.path.join(tmp_dir, bib, bib + '_coordinate.csv')
            with open(annotation_file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                image_meta += self._parse_coord_csv(reader, tmp_dir, bib_id=bib)
        return image_meta

    def _parse_coord_csv(self, csv_reader, tmp_dir, bib_id):
        """Parse a CSV file containing character coordinate info.

        Extracts bbox information (top_left_x, top_right_x,
        width, height) for input images, depending on image scope.

        Args:
            csv_reader:
            tmp_dir:
            bib_id:

        Returns:
            A list of Image namedtuples.

        """
        header = next(csv_reader)
        if self.image_scope == 'full':
            # Full images
            result = []
            unicodes = []
            row = next(csv_reader)
            image_id = row[1]
            unicodes.append(row[0])
            for row in csv_reader:
                if row[1] == image_id:
                    unicodes.append(row[0])
                else:
                    result.append(Image(
                        os.path.join(tmp_dir,
                                     bib_id,
                                     'images',
                                     image_id + '.jpg'),
                        None, None, None, None,
                        unicodes
                    ))
                    image_id = row[1]
                    unicodes = [row[0]]
            # Add last image
            result.append(Image(
                os.path.join(tmp_dir, bib_id, 'images', image_id + '.jpg'),
                None, None, None, None,
                unicodes
            ))
        else:
            # Single characters
            result = [
                Image(
                    os.path.join(tmp_dir, bib_id, 'images', row[1] + '.jpg'),
                    int(row[2]), # X
                    int(row[3]), # Y
                    int(row[6]), # W
                    int(row[7]), # H
                    [row[0]], # label
                ) for row in csv_reader
            ]

            if self.image_scope == 'seq':
                sequences = []
                # Within-line crop around character sequence
                # of length self.num_targets
                step = 1 if self.overlapping else self.num_targets
                for i in xrange(0, len(result) - (self.num_targets - 1), step):
                    xs, ys, ws, hs, labels = [], [], [], [], []
                    for j in range(self.num_targets):
                        xs.append(result[i+j].x)
                        ys.append(result[i+j].y)
                        ws.append(result[i+j].w)
                        hs.append(result[i+j].h)
                        labels += result[i+j].label
                        if result[i+j].y < ys[0] or (
                                result[i+j].filename != result[i].filename):
                            break
                    else:
                        h = (ys[-1] + hs[-1]) - ys[0]
                        sequences.append(Image(
                            result[i].filename,
                            min(xs), min(ys), max(ws), h, labels
                        ))
                result = sequences
        return result

@registry.register_problem()
class OcrPmjtChar(OcrPmjtProblem):

    @property
    def image_scope(self):
        return 'single_char'

    @property
    def image_shape(self):
        return (64, 64)

    def preprocess_example(self, example, mode, hparams):
        example = super(OcrPmjtChar, self).preprocess_example(
            example, mode, hparams)
        example["inputs"].set_shape([64,64,1])
        example["targets"].set_shape([1])
        return example

@registry.register_problem()
class OcrPmjtSeq(OcrPmjtProblem):

    @property
    def image_scope(self):
        return 'sequence'

    @property
    def image_shape(self):
        return (None, 64)

    @property
    def num_targets(self):
        return 3

    @property
    def overlapping(self):
        return False

@registry.register_problem()
class OcrPmjtFull(OcrPmjtProblem):

    @property
    def image_scope(self):
        return 'full'

    @property
    def image_shape(self):
        return (400, 250)