import os
import sys
import six
import json
import gzip
import tensorflow as tf
import logging
from src_nq.create_examples import NqExample

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NqDataset(object):
    def __init__(self, args, input_file, is_training):
        if not is_training:
            self.examples = self.read_nq_examples(input_path=input_file)

        prefix = "cached_{0}_{1}_{2}".format(str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        prefix = os.path.join(args.feature_path, prefix)

        cached_path = os.path.join(prefix, os.path.split(input_file)[1] + ".pkl")
        self.features = self.read_nq_features(cached_path, is_training)

    @staticmethod
    def read_nq_examples(input_path):
        logging.info("Reading examples from {}.".format(input_path))
        examples = []
        with gzip.GzipFile(fileobj=tf.gfile.GFile(input_path, "rb")) as fi:
            for line in fi:
                if not isinstance(line, six.text_type):
                    line = line.decode("utf-8")
                json_example = json.loads(line)

                example_id = json_example["example_id"]
                la_candidates = json_example["long_answer_candidates"]

                example = NqExample(example_id=example_id,
                                    question_tokens=None,
                                    doc_tokens=None,
                                    la_candidates=la_candidates,
                                    annotation=None)
                examples.append(example)
        return examples

    @staticmethod
    def read_nq_features(cached_path, is_training=False):
        if not os.path.exists(cached_path):
            logging.info("{} doesn't exists.".format(cached_path))
            exit(0)
        logging.info("Reading features from {}.".format(cached_path))
        with open(cached_path, "rb") as reader:
            features = pickle.load(reader)

        for i, feature in enumerate(features):
            feature.unique_id = i
        return features
