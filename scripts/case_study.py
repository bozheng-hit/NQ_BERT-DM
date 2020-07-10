import sys
import argparse
import tensorflow as tf
from tqdm import tqdm
import json
import random
import sys
from glob import glob
import gzip
from tools.nq_eval_tools import *
from tools.nq_eval_utils import *
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


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
                 annotation,
                 doc_url):
        self.example_id = example_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.la_candidates = la_candidates
        self.annotation = annotation
        self.doc_url = doc_url


def read_examples_from_one_split(gzipped_input_file):
    """Read annotation from one split of file."""
    if isinstance(gzipped_input_file, str):
        gzipped_input_file = open(gzipped_input_file, "rb")
    logging.info('parsing %s ..... ', gzipped_input_file.name)
    nq_example_dict = {}
    with GzipFile(fileobj=gzipped_input_file) as input_file:
        for line in input_file:
            json_example = json.loads(line)

            question_tokens = json_example["question_tokens"]
            doc_tokens = json_example["document_tokens"]
            annotation = json_example["annotations"]
            example_id = json_example["example_id"]
            la_candidates = json_example["long_answer_candidates"]
            doc_url = json_example["document_url"]

            nq_example_dict[example_id] = NqExample(example_id=example_id,
                                                    question_tokens=question_tokens,
                                                    doc_tokens=doc_tokens,
                                                    la_candidates=la_candidates,
                                                    annotation=annotation,
                                                    doc_url=doc_url)

    return nq_example_dict


def read_nq_examples(path_name, n_threads=10):
    """Read annotations with real multiple processes."""

    input_paths = glob.glob(path_name)
    pool = multiprocessing.Pool(n_threads)
    try:
        dict_list = pool.map(read_examples_from_one_split, input_paths)
    finally:
        pool.close()
        pool.join()

    final_dict = {}
    for single_dict in dict_list:
        final_dict.update(single_dict)

    return final_dict


def read_nq_annotation(path_name, n_threads=10):
    """Read annotations with real multiple processes."""

    input_paths = glob.glob(path_name)
    pool = multiprocessing.Pool(n_threads)
    try:
        dict_list = pool.map(read_annotation_from_one_split, input_paths)
    finally:
        pool.close()
        pool.join()

    final_dict = {}
    for single_dict in dict_list:
        final_dict.update(single_dict)

    return final_dict


def read_nbest_prediction_json(nbest_predictions_path):
    logging.info('Reading nbest predictions from file: %s', format(nbest_predictions_path))
    with open(nbest_predictions_path, 'r') as f:
        predictions = json.loads(f.read())

    nq_nbest_pred_dict = {}
    for nbest_predictions in predictions['predictions']:
        pred_items = []
        for single_prediction in nbest_predictions:
            if 'long_answer' in single_prediction:
                long_span = Span(single_prediction['long_answer']['start_byte'],
                                 single_prediction['long_answer']['end_byte'],
                                 single_prediction['long_answer']['start_token'],
                                 single_prediction['long_answer']['end_token'])
            else:
                long_span = Span(-1, -1, -1, -1)  # Span is null if not presented.

            short_span_list = []
            if 'short_answers' in single_prediction:
                for short_item in single_prediction['short_answers']:
                    short_span_list.append(
                        Span(short_item['start_byte'], short_item['end_byte'],
                             short_item['start_token'], short_item['end_token']))

            yes_no_answer = 'none'
            if 'yes_no_answer' in single_prediction:
                yes_no_answer = single_prediction['yes_no_answer'].lower()
                if yes_no_answer not in ['yes', 'no', 'none']:
                    raise ValueError('Invalid yes_no_answer value in prediction')

                if yes_no_answer != 'none' and not is_null_span_list(short_span_list):
                    raise ValueError('yes/no prediction and short answers cannot coexist.')

            pred_items.append(NQLabel(
                example_id=single_prediction['example_id'],
                long_answer_span=long_span,
                short_answer_span_list=short_span_list,
                yes_no_answer=yes_no_answer,
                long_score=single_prediction['long_answer_score'],
                short_score=single_prediction['short_answers_score']))

        nq_nbest_pred_dict[nbest_predictions[0]['example_id']] = pred_items

    return nq_nbest_pred_dict


def get_context(nq_example, span):
    tokens = [nq_example.doc_tokens[i]['token'] for i in range(span.start_token_idx, span.end_token_idx)]
    context = ' '.join(tokens)
    return context


class Stats(object):
    def __init__(self, l):
        self.gold_has_answer, self.pred_has_answer, self.is_correct, self.score = l


def get_nbest_recall(nq_gold_dict, nq_nbest_pred_dict, nq_example_dict, output_dir):
    example_ids = list(nq_nbest_pred_dict.keys())
    n_has_long_answer = 0
    n_has_short_answer = 0

    n_correct_long_answer = [0] * 20
    n_correct_short_answer = [0] * 20

    for example_id in example_ids:
        example = nq_example_dict[example_id]
        found_long_correct = -1
        found_short_correct = -1
        gold_item = nq_gold_dict[example_id]

        for i, pred_item in enumerate(nq_nbest_pred_dict[example_id]):
            long_stats = Stats(score_long_answer(gold_item, pred_item))
            short_stats = Stats(score_short_answer(gold_item, pred_item))

            if i == 0 and long_stats.gold_has_answer:
                n_has_long_answer += 1
            if i == 0 and short_stats.gold_has_answer:
                n_has_short_answer += 1

            if long_stats.gold_has_answer and long_stats.pred_has_answer and long_stats.is_correct:
                if found_long_correct < 0:
                    found_long_correct = i
                    for j in range(i, 20):
                        n_correct_long_answer[j] += 1

            if short_stats.gold_has_answer and short_stats.pred_has_answer and short_stats.is_correct:
                if found_short_correct < 0:
                    found_short_correct = i
                    for j in range(i, 20):
                        n_correct_short_answer[j] += 1

    print("n_has_long_answer: {}".format(n_has_long_answer))
    print("n_has_short_answer: {}".format(n_has_short_answer))

    for i in range(20):
        print("TOP " + str(i) + " Recall:", n_correct_long_answer[i] / n_has_long_answer * 100)

    for i in range(20):
        print("TOP " + str(i) + " Recall:", n_correct_short_answer[i] / n_has_short_answer * 100)


def print_item_context(example, item, fout, type):
    fout.write("=" * 20 + " " + type + " " + "=" * 20 + '\n')
    fout.write("Long Answer: Score {}\t Span: {}\t Context: {}\n".format(
        item.long_score,
        item.long_answer_span,
        get_context(example, item.long_answer_span)))

    fout.write("Short Answer Score: {} \n".format(
        item.short_score))
    for i, short_answer in enumerate(item.short_answer_span_list):
        fout.write("Short Answer {}: Span: {}\t Context: {}\n".format(i, short_answer,
                                                                      get_context(example, short_answer)))


def analysis(nq_gold_dict, nq_pred_dict_by_short, nq_pred_dict_by_long, nq_example_dict, output_dir,
             long_threshold_by_short, short_threshold_by_short, long_threshold_by_long, short_threshold_by_long):
    fout_case_1 = open(os.path.join(output_dir, "case_study_1.txt"), "w")
    fout_case_2 = open(os.path.join(output_dir, "case_study_2.txt"), "w")

    has_long_answer_correct_by_short = 0
    no_long_answer_correct_by_short = 0
    has_long_answer_correct_by_long = 0
    no_long_answer_correct_by_long = 0

    has_short_answer_correct_by_short = 0
    no_short_answer_correct_by_short = 0
    has_short_answer_correct_by_long = 0
    no_short_answer_correct_by_long = 0

    example_ids = list(nq_pred_dict_by_short.keys())
    for example_id in example_ids:
        prediction_by_short = nq_pred_dict_by_short[example_id]
        prediction_by_long = nq_pred_dict_by_long[example_id]
        gold_item = nq_gold_dict[example_id]

        example = nq_example_dict[example_id]

        long_stats_by_short = Stats(score_long_answer(gold_item, prediction_by_short))
        long_stats_by_long = Stats(score_long_answer(gold_item, prediction_by_long))

        short_stats_by_short = Stats(score_short_answer(gold_item, prediction_by_short))
        short_stats_by_long = Stats(score_short_answer(gold_item, prediction_by_long))

        if long_stats_by_short.is_correct:
            if long_stats_by_short.gold_has_answer:
                has_long_answer_correct_by_short += 1
            else:
                no_long_answer_correct_by_short += 1

        if long_stats_by_long.is_correct:
            if long_stats_by_long.gold_has_answer:
                has_long_answer_correct_by_long += 1
            else:
                no_long_answer_correct_by_long += 1

        if short_stats_by_short.is_correct:
            if short_stats_by_short.gold_has_answer:
                has_short_answer_correct_by_short += 1
            else:
                no_short_answer_correct_by_short += 1

        if short_stats_by_long.is_correct:
            if short_stats_by_long.gold_has_answer:
                has_short_answer_correct_by_long += 1
            else:
                no_short_answer_correct_by_long += 1

        if not long_stats_by_short.is_correct and long_stats_by_long.is_correct:
            # print("123")
            if not short_stats_by_short.is_correct and short_stats_by_long.is_correct:
                # print("456")
                fout_case_1.write("Example id: {}\n".format(example_id))
                fout_case_1.write("Document Url: {}\n".format(example.doc_url))
                fout_case_1.write("Question: {}\n".format(' '.join(example.question_tokens)))
                for i in range(5):
                    print_item_context(example, gold_item[i], fout_case_1, "gold_{}".format(i))

                print_item_context(example, prediction_by_short, fout_case_1, "wrong_prediction")
                print_item_context(example, prediction_by_long, fout_case_1, "correct_prediction")
                fout_case_1.write("\n")

        if not long_stats_by_long.is_correct and long_stats_by_short.is_correct:
            # print("78")
            if not short_stats_by_long.is_correct and short_stats_by_short.is_correct:
                # print("9")
                fout_case_2.write("Example id: {}\n".format(example_id))
                fout_case_2.write("Document Url: {}\n".format(example.doc_url))
                fout_case_2.write("Question: {}\n".format(' '.join(example.question_tokens)))
                for i in range(5):
                    print_item_context(example, gold_item[i], fout_case_2, "gold_{}".format(i))

                print_item_context(example, prediction_by_long, fout_case_2, "wrong_prediction")
                print_item_context(example, prediction_by_short, fout_case_2, "correct_prediction")
                fout_case_2.write("\n")

    print("has_long_answer_correct_by_short:", has_long_answer_correct_by_short)
    print("no_long_answer_correct_by_short:", no_long_answer_correct_by_short)
    print("has_long_answer_correct_by_long:", has_long_answer_correct_by_long)
    print("no_long_answer_correct_by_long:", no_long_answer_correct_by_long)
    print("has_short_answer_correct_by_short:", has_short_answer_correct_by_short)
    print("no_short_answer_correct_by_short:", no_short_answer_correct_by_short)
    print("has_short_answer_correct_by_long:", has_short_answer_correct_by_long)
    print("no_short_answer_correct_by_long:", no_short_answer_correct_by_long)


def analysis_correct(nq_gold_dict, nq_pred_dict_by_short, nq_pred_dict_by_long, nq_example_dict, output_dir,
                     long_threshold_by_short, short_threshold_by_short, long_threshold_by_long,
                     short_threshold_by_long):
    n_bad_context = 0
    n_corrected = 0

    has_long_answer_correct_by_short = 0
    no_long_answer_correct_by_short = 0
    has_long_answer_correct_by_long = 0
    no_long_answer_correct_by_long = 0

    has_short_answer_correct_by_short = 0
    no_short_answer_correct_by_short = 0
    has_short_answer_correct_by_long = 0
    no_short_answer_correct_by_long = 0

    example_ids = list(nq_pred_dict_by_short.keys())
    for example_id in example_ids:
        prediction_by_short = nq_pred_dict_by_short[example_id]
        prediction_by_long = nq_pred_dict_by_long[example_id]
        gold_item = nq_gold_dict[example_id]

        example = nq_example_dict[example_id]

        long_stats_by_short = Stats(score_long_answer(gold_item, prediction_by_short))
        long_stats_by_long = Stats(score_long_answer(gold_item, prediction_by_long))

        short_stats_by_short = Stats(score_short_answer(gold_item, prediction_by_short))
        short_stats_by_long = Stats(score_short_answer(gold_item, prediction_by_long))

        if long_stats_by_short.is_correct:
            if long_stats_by_short.gold_has_answer:
                has_long_answer_correct_by_short += 1
            else:
                no_long_answer_correct_by_short += 1

        if long_stats_by_long.is_correct:
            if long_stats_by_long.gold_has_answer:
                has_long_answer_correct_by_long += 1
            else:
                no_long_answer_correct_by_long += 1

        if short_stats_by_short.is_correct:
            if short_stats_by_short.gold_has_answer:
                has_short_answer_correct_by_short += 1
            else:
                no_short_answer_correct_by_short += 1

        if short_stats_by_long.is_correct:
            if short_stats_by_long.gold_has_answer:
                has_short_answer_correct_by_long += 1
            else:
                no_short_answer_correct_by_long += 1

        if not short_stats_by_short.is_correct and short_stats_by_short.gold_has_answer:
            found = False
            for i in range(5):
                if len(gold_item[i].short_answer_span_list) == 1 and \
                    get_context(example, prediction_by_short.short_answer_span_list[0]) == \
                    get_context(example, gold_item[i].short_answer_span_list[0]):
                    found = True

            if found:
                n_bad_context += 1
                if short_stats_by_long.is_correct and \
                    get_context(example, prediction_by_short.short_answer_span_list[0]) == \
                    get_context(example, prediction_by_long.short_answer_span_list[0]):
                    n_corrected += 1

    print("n_bad_context: ", n_bad_context)
    print("n_corrected: ", n_corrected)


def analysis_condition(nq_gold_dict, nq_pred_dict, nq_example_dict, long_threshold, short_threshold):
    n_bad_context = 0
    n_corrected = 0
    example_ids = list(nq_pred_dict.keys())

    n_long_case1 = 0
    n_long_case2 = 0
    n_long_case3 = 0
    n_long_case4 = 0
    n_long_case5 = 0

    n_short_case1 = 0
    n_short_case2 = 0
    n_short_case3 = 0
    n_short_case4 = 0
    n_short_case5 = 0

    for example_id in example_ids:
        pred_item = nq_pred_dict[example_id]
        gold_item = nq_gold_dict[example_id]

        example = nq_example_dict[example_id]

        long_stats = Stats(score_long_answer(gold_item, pred_item))
        short_stats = Stats(score_short_answer(gold_item, pred_item))

        if long_stats.gold_has_answer and long_stats.score >= long_threshold and long_stats.is_correct:
            n_long_case1 += 1
        if not long_stats.gold_has_answer and long_stats.score < long_threshold:
            n_long_case2 += 1
        if long_stats.gold_has_answer and long_stats.score >= long_threshold and not long_stats.is_correct:
            n_long_case3 += 1
        if long_stats.gold_has_answer and long_stats.score < long_threshold:
            n_long_case4 += 1
        if not long_stats.gold_has_answer and long_stats.score >= long_threshold:
            n_long_case5 += 1

        if short_stats.gold_has_answer and short_stats.score >= short_threshold and short_stats.is_correct:
            n_short_case1 += 1
        if not short_stats.gold_has_answer and short_stats.score < short_threshold:
            n_short_case2 += 1
        if short_stats.gold_has_answer and short_stats.score >= short_threshold and not short_stats.is_correct:
            n_short_case3 += 1
        if short_stats.gold_has_answer and short_stats.score < short_threshold:
            n_short_case4 += 1
        if not short_stats.gold_has_answer and short_stats.score >= short_threshold:
            n_short_case5 += 1

    print(n_long_case1, n_long_case2, n_long_case3, n_long_case4, n_long_case5)
    print(n_short_case1, n_short_case2, n_short_case3, n_short_case4, n_short_case5)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--gold_pattern", required=True, type=str, help="path to dataset.")
    cmd.add_argument("--predictions_path", required=True, type=str, help="")
    args = cmd.parse_args(sys.argv[1:])

    random.seed(0)

    nq_example_dict = read_nq_examples(args.gold_pattern, n_threads=16)

    nq_gold_dict = read_nq_annotation(
        args.gold_pattern, n_threads=16)

    nq_pred_dict = util.read_prediction_json(args.predictions_path)

    long_answer_stats, short_answer_stats = score_answers(nq_gold_dict, nq_pred_dict)

    metrics = get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)

    metrics = my_analysis(metrics, long_answer_stats, short_answer_stats)

    print(metrics)

    long_threshold = metrics["long-best-threshold"]
    short_threshold = metrics["short-best-threshold"]

    analysis_condition(nq_gold_dict, nq_pred_dict, nq_example_dict, long_threshold,
                       short_threshold)
