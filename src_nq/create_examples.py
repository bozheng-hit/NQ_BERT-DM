# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import enum
import os
import random
import sys
sys.path.append("/nq_model/")

from io import open

import numpy as np
import torch
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer
from glob import glob
import tensorflow as tf
import gzip
import multiprocessing

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import six

from spacy.lang.en import English
from modules.graph_encoder import NodePosition, Graph, EdgeType, get_edge_position

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


class NqExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 question_tokens,
                 doc_tokens,
                 la_candidates,
                 annotation):
        self.example_id = example_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.la_candidates = la_candidates
        self.annotation = annotation


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 graph,
                 answer_type=None,
                 start_positions=None,
                 end_positions=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.graph = graph
        self.answer_type = answer_type
        self.start_positions = start_positions
        self.end_positions = end_positions


class NodeInfo(object):
    def __init__(self,
                 start_position,
                 end_position,
                 node_idx):
        self.start_position = start_position
        self.end_position = end_position
        self.node_idx = node_idx


def build_graph(doc_tree, seq_len, la_start_position, la_end_position, sa_start_position, sa_end_position):
    graph = Graph()
    doc_node_idx = NodePosition.MAX_TOKEN + NodePosition.MAX_SENTENCE + NodePosition.MAX_PARAGRAPH
    graph.add_node(doc_node_idx, -1)
    doc_node = NodeInfo(node_idx=doc_node_idx, start_position=0, end_position=0)
    para_nodes = []
    sent_nodes = []
    token_nodes = []

    for para_idx, (candidate_idx, paragraph) in enumerate(doc_tree):
        if len(para_nodes) < NodePosition.MAX_PARAGRAPH:
            para_node_idx = NodePosition.MAX_TOKEN + NodePosition.MAX_SENTENCE + len(para_nodes)
            graph.add_node(para_node_idx, candidate_idx)
            para_nodes.append(
                NodeInfo(node_idx=para_node_idx, start_position=len(token_nodes), end_position=len(token_nodes)))
        else:
            para_node_idx = -1

        graph.add_edge(para_node_idx, doc_node_idx, edge_type=EdgeType.PARAGRAPH_TO_DOCUMENT,
                       edge_pos=get_edge_position(EdgeType.PARAGRAPH_TO_DOCUMENT, len(para_nodes) - 1))
        graph.add_edge(doc_node_idx, para_node_idx, edge_type=EdgeType.DOCUMENT_TO_PARAGRAPH,
                       edge_pos=get_edge_position(EdgeType.DOCUMENT_TO_PARAGRAPH, len(para_nodes) - 1))

        for sent_idx, (sentence_idx, sentence) in enumerate(paragraph):
            if len(sent_nodes) < NodePosition.MAX_SENTENCE:
                sent_node_idx = NodePosition.MAX_TOKEN + len(sent_nodes)
                graph.add_node(sent_node_idx, sentence_idx)
                sent_nodes.append(
                    NodeInfo(node_idx=sent_node_idx, start_position=len(token_nodes), end_position=len(token_nodes)))
            else:
                sent_node_idx = -1

            graph.add_edge(sent_node_idx, para_node_idx, edge_type=EdgeType.SENTENCE_TO_PARAGRAPH,
                           edge_pos=get_edge_position(EdgeType.SENTENCE_TO_PARAGRAPH, sent_idx))
            graph.add_edge(para_node_idx, sent_node_idx, edge_type=EdgeType.PARAGRAPH_TO_SENTENCE,
                           edge_pos=get_edge_position(EdgeType.PARAGRAPH_TO_SENTENCE, sent_idx))

            graph.add_edge(sent_node_idx, doc_node_idx, edge_type=EdgeType.SENTENCE_TO_DOCUMENT,
                           edge_pos=get_edge_position(EdgeType.SENTENCE_TO_DOCUMENT, len(sent_nodes) - 1))
            graph.add_edge(doc_node_idx, sent_node_idx, edge_type=EdgeType.DOCUMENT_TO_SENTENCE,
                           edge_pos=get_edge_position(EdgeType.DOCUMENT_TO_SENTENCE, len(sent_nodes) - 1))

            for token_idx, (orig_tok_idx, _) in enumerate(sentence):
                token_node_idx = len(token_nodes)
                graph.add_node(token_node_idx, orig_tok_idx)
                token_nodes.append(
                    NodeInfo(node_idx=token_node_idx, start_position=len(token_nodes), end_position=len(token_nodes)))

                graph.add_edge(token_node_idx, sent_node_idx, edge_type=EdgeType.TOKEN_TO_SENTENCE,
                               edge_pos=get_edge_position(EdgeType.TOKEN_TO_SENTENCE, token_idx))
                graph.add_edge(sent_node_idx, token_node_idx, edge_type=EdgeType.SENTENCE_TO_TOKEN,
                               edge_pos=get_edge_position(EdgeType.SENTENCE_TO_TOKEN, token_idx))

                graph.add_edge(token_node_idx, para_node_idx, edge_type=EdgeType.TOKEN_TO_PARAGRAPH,
                               edge_pos=get_edge_position(EdgeType.TOKEN_TO_PARAGRAPH,
                                                          len(token_nodes) - 1 - para_nodes[-1].start_position))
                graph.add_edge(para_node_idx, token_node_idx, edge_type=EdgeType.PARAGRAPH_TO_TOKEN,
                               edge_pos=get_edge_position(EdgeType.PARAGRAPH_TO_TOKEN,
                                                          len(token_nodes) - 1 - para_nodes[-1].start_position))

                graph.add_edge(token_node_idx, doc_node_idx, edge_type=EdgeType.TOKEN_TO_DOCUMENT,
                               edge_pos=get_edge_position(EdgeType.TOKEN_TO_DOCUMENT, len(token_nodes) - 1))
                graph.add_edge(doc_node_idx, token_node_idx, edge_type=EdgeType.DOCUMENT_TO_TOKEN,
                               edge_pos=get_edge_position(EdgeType.DOCUMENT_TO_TOKEN, len(token_nodes) - 1))

            if sent_node_idx >= 0:
                sent_nodes[-1].end_position = len(token_nodes) - 1
        if para_node_idx >= 0:
            para_nodes[-1].end_position = len(token_nodes) - 1
    doc_node.end_position = len(token_nodes) - 1

    assert len(token_nodes) == seq_len

    if la_start_position is not None:
        start_positions = [-1] * 3
        end_positions = [-1] * 3
    else:
        start_positions = None
        end_positions = None

    for token_idx, token_node in enumerate(token_nodes):
        if start_positions is not None:
            if token_node.start_position <= sa_start_position <= token_node.end_position:
                start_positions[0] = token_idx
            if token_node.start_position <= sa_end_position <= token_node.end_position:
                end_positions[0] = token_idx

    for sent_idx, sent_node in enumerate(sent_nodes):
        if start_positions is not None:
            if sent_node.start_position <= sa_start_position <= sent_node.end_position:
                start_positions[1] = sent_idx
            if sent_node.start_position <= sa_end_position <= sent_node.end_position:
                end_positions[1] = sent_idx

    for para_idx, para_node in enumerate(para_nodes):
        if start_positions is not None:
            if para_node.start_position <= la_start_position <= para_node.end_position:
                start_positions[2] = para_idx
            if para_node.start_position <= la_end_position <= para_node.end_position:
                end_positions[2] = para_idx

    if start_positions is not None:
        assert start_positions[2] == end_positions[2]
        assert start_positions[0] == sa_start_position
        assert end_positions[0] == sa_end_position

    return graph, start_positions, end_positions


def get_doc_tree(is_sentence_end, is_paragraph_end, orig_tok_idx, candidate_idx):
    doc_len = len(is_sentence_end)
    document = []
    paragraph = []
    sentence = []
    cur_candidate_idx = -1
    for i in range(doc_len):
        sentence.append((orig_tok_idx[i], i))
        cur_candidate_idx = max(cur_candidate_idx, candidate_idx[i])

        if is_sentence_end[i]:
            paragraph.append((-1, sentence))
            sentence = []
        if is_paragraph_end[i]:
            assert len(sentence) == 0
            document.append((cur_candidate_idx, paragraph))
            paragraph = []
            cur_candidate_idx = -1
    assert len(sentence) == 0
    assert len(paragraph) == 0
    return document


def get_candidate_type(token, counts, max_position):
    if token == "<Table>":
        counts["Table"] += 1
        return min(counts["Table"], max_position) - 1
    elif token == "<P>":
        counts["Paragraph"] += 1
        return min(counts["Paragraph"] + max_position, max_position * 2) - 1
    elif token in ("<Ul>", "<Dl>", "<Ol>"):
        counts["List"] += 1
        return min(counts["List"] + max_position * 2, max_position * 3) - 1
    elif token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        counts["Other"] += 1
        return min(counts["Other"] + max_position * 3, max_position * 4) - 1
    else:
        logging.info("Unknoww candidate type found: %s", token)
        counts["Other"] += 1
        return min(counts["Other"] + max_position * 3, max_position * 4) - 1


def convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    """Loads a data file into a list of `InputBatch`s."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = []
        for token in example.question_tokens:
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                query_tokens.append(sub_token)

        if len(query_tokens) > args.max_query_length:
            query_tokens = query_tokens[-args.max_query_length:]

        counts = {"Table": 0, "Paragraph": 0, "List": 0, "Other": 0}

        sorted_la_candidates = sorted(enumerate(example.la_candidates), key=lambda x: x[1]["start_token"])

        tok_is_sentence_end = []
        tok_is_paragraph_end = []
        tok_candidate_idx = []
        tok_to_orig_index = []
        orig_to_tok_index = [-1] * len(example.doc_tokens)
        orig_to_tok_end_index = [-1] * len(example.doc_tokens)
        all_doc_tokens = []

        for candidate_idx, candidate in sorted_la_candidates:
            if not candidate["top_level"]:
                continue
            start_index = candidate["start_token"]
            end_index = candidate["end_token"]

            context_tokens = []
            context_orig_idx = []

            for tok_idx in range(start_index, end_index):
                token_item = example.doc_tokens[tok_idx]
                if not token_item["html_token"]:
                    context_tokens.append(token_item["token"].replace(" ", ""))
                    if context_tokens[-1] == "":
                        context_tokens = context_tokens[:-1]
                        continue
                    context_orig_idx.append(tok_idx)

            if len(context_tokens) == 0:
                continue

            all_doc_tokens.append(
                "[unused{}]".format(
                    get_candidate_type(example.doc_tokens[start_index]["token"], counts, args.max_position)))
            tok_to_orig_index.append(-1)
            tok_is_sentence_end.append(False)
            tok_is_paragraph_end.append(False)
            tok_candidate_idx.append(candidate_idx)

            paragraph_len = 0
            context = " ".join(context_tokens)
            context_sentences = nlp(context).sents
            context_idx = 0
            orig_to_tok_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
            for sentence in context_sentences:
                sentence_len = 0
                for token in sentence.string.strip().split():
                    if len(context_tokens[context_idx]) == 0:
                        orig_to_tok_end_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
                        context_idx += 1
                        orig_to_tok_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
                    assert context_tokens[context_idx][:len(token)] == token
                    context_tokens[context_idx] = context_tokens[context_idx][len(token):]

                    sub_tokens = tokenizer.tokenize(token)

                    all_doc_tokens += sub_tokens
                    tok_to_orig_index += [context_orig_idx[context_idx]] * len(sub_tokens)
                    sentence_len += len(sub_tokens)
                    paragraph_len += len(sub_tokens)
                if sentence_len > 0:
                    tok_is_sentence_end += [False] * (sentence_len - 1) + [True]

            orig_to_tok_end_index[context_orig_idx[context_idx]] = len(all_doc_tokens)
            assert context_idx + 1 == len(context_tokens)
            assert context_tokens[context_idx] == ""

            tok_is_paragraph_end += [False] * (paragraph_len - 1) + [True]
            tok_candidate_idx += [candidate_idx] * paragraph_len

        assert len(tok_is_sentence_end) == len(tok_is_paragraph_end)

        la_tok_start_position = None
        la_tok_end_position = None

        sa_tok_start_position = None
        sa_tok_end_position = None

        example_answer_type = None

        if is_training:
            annotation = example.annotation[0]
            if annotation["long_answer"]["start_token"] != -1:
                la_start_token = annotation["long_answer"]["start_token"]
                la_end_token = annotation["long_answer"]["end_token"] - 1
                while orig_to_tok_index[la_start_token] == -1:
                    la_start_token += 1
                while orig_to_tok_end_index[la_end_token] == -1:
                    la_end_token -= 1

                assert la_start_token <= la_end_token

                la_tok_start_position = orig_to_tok_index[la_start_token]
                la_tok_end_position = orig_to_tok_end_index[la_end_token] - 1

                if annotation["yes_no_answer"].lower() == "none":
                    example_answer_type = AnswerType.LONG
                elif annotation["yes_no_answer"].lower() == "yes":
                    example_answer_type = AnswerType.YES
                elif annotation["yes_no_answer"].lower() == "no":
                    example_answer_type = AnswerType.NO
                else:
                    assert False

                if "short_answers" in annotation and len(annotation["short_answers"]) > 0:
                    sa_tok_start_position = len(all_doc_tokens) + 1
                    sa_tok_end_position = -1
                    example_answer_type = AnswerType.SHORT
                    for short_answer in annotation["short_answers"]:
                        sa_start_token = short_answer["start_token"]
                        sa_end_token = short_answer["end_token"] - 1

                        sa_tok_start_position = min(sa_tok_start_position, orig_to_tok_index[sa_start_token])
                        sa_tok_end_position = max(sa_tok_end_position, orig_to_tok_end_index[sa_end_token] - 1)
                        assert sa_tok_start_position >= 0 and sa_tok_end_position >= 0
                    assert la_tok_start_position <= sa_tok_start_position and la_tok_end_position >= sa_tok_end_position
                    la_tok_start_position = sa_tok_start_position
                    la_tok_end_position = sa_tok_end_position
                else:
                    sa_tok_start_position = -1
                    sa_tok_end_position = -1
            else:
                example_answer_type = AnswerType.UNKNOWN
                la_tok_start_position = -1
                la_tok_end_position = -1
                sa_tok_start_position = -1
                sa_tok_end_position = -1

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, args.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            is_sentence_end = [True] + [False] * (len(tokens) - 3) + [True] + [False]
            is_paragraph_end = [True] + [False] * (len(tokens) - 3) + [True] + [False]
            orig_tok_idx = [-1] * len(tokens)
            candidate_idx = [-1] * len(tokens)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)

                is_sentence_end.append(tok_is_sentence_end[split_token_index])
                is_paragraph_end.append(tok_is_paragraph_end[split_token_index])
                orig_tok_idx.append(tok_to_orig_index[split_token_index])
                candidate_idx.append(tok_candidate_idx[split_token_index])

            is_sentence_end[-1] = False
            is_paragraph_end[-1] = False

            tokens.append("[SEP]")
            segment_ids.append(1)

            is_sentence_end.append(True)
            is_paragraph_end.append(True)
            orig_tok_idx.append(-1)
            candidate_idx.append(-1)

            assert len(orig_tok_idx) == len(tokens)
            assert len(is_sentence_end) == len(tokens)
            assert len(is_paragraph_end) == len(tokens)
            assert len(candidate_idx) == len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < args.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length

            la_start_position = None
            la_end_position = None
            sa_start_position = None
            sa_end_position = None
            answer_type = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (la_tok_start_position >= doc_start and
                        la_tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    if random.random() > args.include_unknowns:
                        continue
                    answer_type = AnswerType.UNKNOWN
                    la_start_position = 0
                    la_end_position = 0
                    sa_start_position = 0
                    sa_end_position = 0
                else:
                    answer_type = example_answer_type
                    doc_offset = len(query_tokens) + 2
                    la_start_position = la_tok_start_position - doc_start + doc_offset
                    la_end_position = la_tok_end_position - doc_start + doc_offset

                    if not (sa_tok_start_position >= doc_start and sa_tok_end_position <= doc_end):
                        sa_start_position = -1
                        sa_end_position = -1
                    else:
                        sa_start_position = sa_tok_start_position - doc_start + doc_offset
                        sa_end_position = sa_tok_end_position - doc_start + doc_offset

                assert la_start_position <= la_end_position
                assert sa_start_position <= sa_end_position

            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and answer_type != AnswerType.UNKNOWN:
                    answer_text = " ".join(tokens[la_start_position:(la_end_position + 1)])
                    logger.info("la_start_position: %d" % (la_start_position))
                    logger.info("la_end_position: %d" % (la_end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            doc_tree = get_doc_tree(is_sentence_end, is_paragraph_end, orig_tok_idx, candidate_idx)

            graph, start_positions, end_positions = build_graph(doc_tree, len(is_sentence_end),
                                                                la_start_position, la_end_position,
                                                                sa_start_position, sa_end_position)

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_id=example.example_id,
                    doc_span_index=doc_span_index,
                    tokens=None,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    graph=graph,
                    answer_type=answer_type,
                    start_positions=start_positions,
                    end_positions=end_positions))
            unique_id += 1

    logging.info("  Saving features into cached file {}".format(cached_path))
    with open(cached_path, "wb") as writer:
        pickle.dump(features, writer)

    return cached_path


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def run_convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    pool = []
    p = multiprocessing.Pool(args.num_threads)
    for i in range(args.num_threads):
        start_index = len(examples) // args.num_threads * i
        end_index = len(examples) // args.num_threads * (i + 1)
        if i == args.num_threads - 1:
            end_index = len(examples)
        pool.append(p.apply_async(convert_examples_to_features, args=(
            args, examples[start_index: end_index], tokenizer, is_training, cached_path + ".part" + str(i))))
    p.close()
    p.join()

    features = []
    for i, thread in enumerate(pool):
        cached_path_tmp = thread.get()
        logging.info("Reading thread {} output from {}".format(i, cached_path_tmp))
        with open(cached_path_tmp, "rb") as reader:
            features_tmp = pickle.load(reader)
        os.remove(cached_path_tmp)
        features += features_tmp

    logging.info("  Saving features from into cached file {0}".format(cached_path))
    with open(cached_path, "wb") as writer:
        pickle.dump(features, writer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pattern", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--vocab_file", default=None, type=str, required=True)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--include_unknowns", default=0.03, type=float)
    parser.add_argument("--max_position", default=50, type=int)
    parser.add_argument("--num_threads", default=16, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_num', type=int, default=-1)
    parser.add_argument('--end_num', type=int, default=-1)

    args = parser.parse_args()

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    prefix = "cached_{0}_{1}_{2}".format(str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

    prefix = os.path.join(args.output_dir, prefix)
    os.makedirs(prefix, exist_ok=True)

    for input_path in glob(args.input_pattern):
        num = input_path[-11:-9]
        if args.start_num >= 0 and args.end_num >= 0 and (int(num) > args.end_num or int(num) < args.start_num):
            continue
        cached_path = os.path.join(prefix, os.path.split(input_path)[1] + ".pkl")
        if os.path.exists(cached_path):
            logging.info("{} already exists.".format(cached_path))
            continue
        is_training = True if input_path.find("train") != -1 else False

        examples = []
        with gzip.GzipFile(fileobj=tf.gfile.GFile(input_path, "rb")) as fi:
            logging.info("Reading data from {}.".format(input_path))
            for line in fi:
                if not isinstance(line, six.text_type):
                    line = line.decode("utf-8")
                json_example = json.loads(line)

                question_tokens = json_example["question_tokens"]
                doc_tokens = json_example["document_tokens"]
                annotation = json_example["annotations"]
                example_id = json_example["example_id"]
                la_candidates = json_example["long_answer_candidates"]

                examples.append(
                    NqExample(example_id=example_id,
                              question_tokens=question_tokens,
                              doc_tokens=doc_tokens,
                              la_candidates=la_candidates,
                              annotation=annotation))

        run_convert_examples_to_features(args=args,
                                         examples=examples,
                                         tokenizer=tokenizer,
                                         is_training=is_training,
                                         cached_path=cached_path)


if __name__ == "__main__":
    main()
