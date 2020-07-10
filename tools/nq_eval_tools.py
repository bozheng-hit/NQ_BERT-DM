from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import tools.nq_eval_utils as util


def safe_divide(x, y):
    """Compute x / y, but return 0 if y is zero."""
    if y == 0:
        return 0
    else:
        return x / y


def score_long_answer(gold_label_list, pred_label):
    """Scores a long answer as correct or not.

    1) First decide if there is a gold long answer with LONG_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold long answer.
       b. The prediction span match exactly with *one* of the non-null gold
          long answer span.

    Args:
      gold_label_list: A list of NQLabel, could be None.
      pred_label: A single NQLabel, could be None.

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """
    gold_has_answer = util.gold_has_long_answer(gold_label_list)

    pred_has_answer = pred_label and (
        not pred_label.long_answer_span.is_null_span())

    is_correct = False
    score = pred_label.long_score

    # Both sides are non-null spans.
    if gold_has_answer and pred_has_answer:
        for gold_label in gold_label_list:
            # while the voting results indicate there is an long answer, each
            # annotator might still say there is no long answer.
            if gold_label.long_answer_span.is_null_span():
                continue

            if util.nonnull_span_equal(gold_label.long_answer_span,
                                       pred_label.long_answer_span):
                is_correct = True
                break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_short_answer(gold_label_list, pred_label):
    """Scores a short answer as correct or not.

    1) First decide if there is a gold short answer with SHORT_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold short answer.
       b. The prediction span *set* match exactly with *one* of the non-null gold
          short answer span *set*.

    Args:
      gold_label_list: A list of NQLabel.
      pred_label: A single NQLabel.

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """

    # There is a gold short answer if gold_label_list not empty and non null
    # answers is over the threshold (sum over annotators).
    gold_has_answer = util.gold_has_short_answer(gold_label_list)

    # There is a pred long answer if pred_label is not empty and short answer
    # set is not empty.
    pred_has_answer = pred_label and (
        (not util.is_null_span_list(pred_label.short_answer_span_list)) or
        pred_label.yes_no_answer != 'none')

    is_correct = False
    score = pred_label.short_score

    # Both sides have short answers, which contains yes/no questions.
    if gold_has_answer and pred_has_answer:
        if pred_label.yes_no_answer != 'none':  # System thinks its y/n questions.
            for gold_label in gold_label_list:
                if pred_label.yes_no_answer == gold_label.yes_no_answer:
                    is_correct = True
                    break
        else:
            for gold_label in gold_label_list:
                if util.span_set_equal(gold_label.short_answer_span_list,
                                       pred_label.short_answer_span_list):
                    is_correct = True
                    break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_answers(gold_annotation_dict, pred_dict):
    """Scores all answers for all documents.

    Args:
      gold_annotation_dict: a dict from example id to list of NQLabels.
      pred_dict: a dict from example id to list of NQLabels.

    Returns:
      long_answer_stats: List of scores for long answers.
      short_answer_stats: List of scores for short answers.
    """
    gold_id_set = set(gold_annotation_dict.keys())
    pred_id_set = set(pred_dict.keys())

    if gold_id_set.symmetric_difference(pred_id_set):
        raise ValueError('ERROR: the example ids in gold annotations and example '
                         'ids in the prediction are not equal.')

    long_answer_stats = []
    short_answer_stats = []

    for example_id in gold_id_set:
        gold = gold_annotation_dict[example_id]
        pred = pred_dict[example_id]

        long_answer_stats.append(score_long_answer(gold, pred))
        short_answer_stats.append(score_short_answer(gold, pred))

    # use the 'score' column, which is last
    long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
    short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

    return long_answer_stats, short_answer_stats


def compute_f1(answer_stats, prefix=''):
    """Computes F1, precision, recall for a list of answer scores.

    Args:
      answer_stats: List of per-example scores.
      prefix (''): Prefix to prepend to score dictionary.

    Returns:
      Dictionary mapping string names to scores.
    """

    has_gold, has_pred, is_correct, _ = zip(*answer_stats)
    precision = safe_divide(sum(is_correct), sum(has_pred))
    recall = safe_divide(sum(is_correct), sum(has_gold))
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return OrderedDict({
        prefix + 'n': len(answer_stats),
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall
    })


def compute_final_f1(long_answer_stats, short_answer_stats):
    """Computes overall F1 given long and short answers, ignoring scores.

    Note: this assumes that the answers have been thresholded.

    Arguments:
       long_answer_stats: List of long answer scores.
       short_answer_stats: List of short answer scores.

    Returns:
       Dictionary of name (string) -> score.
    """
    scores = compute_f1(long_answer_stats, prefix='long-answer-')
    scores.update(compute_f1(short_answer_stats, prefix='short-answer-'))
    return scores


def compute_pr_curves(answer_stats, targets=None):
    """Computes PR curve and returns R@P for specific targets.

    The values are computed as follows: find the (precision, recall) point
    with maximum recall and where precision > target.

    Arguments:
      answer_stats: List of statistic tuples from the answer scores.
      targets (None): List of precision thresholds to target.

    Returns:
      List of table with rows: [target, r, p, score].
    """
    total_correct = 0
    total_has_pred = 0
    total_has_gold = 0

    # Count the number of gold annotations.
    for has_gold, _, _, _ in answer_stats:
        total_has_gold += has_gold

    # Keep track of the point of maximum recall for each target.
    max_recall = [0 for _ in targets]
    max_precision = [0 for _ in targets]
    max_scores = [None for _ in targets]

    # Only keep track of unique thresholds in this dictionary.
    scores_to_stats = OrderedDict()

    # Loop through every possible threshold and compute precision + recall.
    for has_gold, has_pred, is_correct, score in answer_stats:
        total_correct += is_correct
        total_has_pred += has_pred

        precision = safe_divide(total_correct, total_has_pred)
        recall = safe_divide(total_correct, total_has_gold)

        # If there are any ties, this will be updated multiple times until the
        # ties are all counted.
        scores_to_stats[score] = [precision, recall]

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = 0.0

    for threshold, (precision, recall) in scores_to_stats.items():
        # Match the thresholds to the find the closest precision above some target.
        for t, target in enumerate(targets):
            if precision >= target and recall > max_recall[t]:
                max_recall[t] = recall
                max_precision[t] = precision
                max_scores[t] = threshold

        # Compute optimal threshold.
        f1 = safe_divide(2 * precision * recall, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = threshold

    return ((best_f1, best_precision, best_recall, best_threshold),
            zip(targets, max_recall, max_precision, max_scores))


def print_r_at_p_table(answer_stats):
    """Pretty prints the R@P table for default targets."""
    opt_result, pr_table = compute_pr_curves(
        answer_stats, targets=[0.5, 0.75, 0.9])
    f1, precision, recall, threshold = opt_result
    print('Optimal threshold: {:.5}'.format(threshold))
    print(' F1     /  P      /  R')
    print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
    for target, recall, precision, row in pr_table:
        if row is None:
            print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={})'.format(
                target, recall, precision, row))
        else:
            print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
                target, recall, precision, row))


def get_metrics_as_dict(gold_path, prediction_path, num_threads=10):
    """Library version of the end-to-end evaluation.

    Arguments:
      gold_path: Path to the gzip JSON data. For multiple files, should be a glob
        pattern (e.g. "/path/to/files-*")
      prediction_path: Path to the JSON prediction data.
      num_threads (10): Number of threads to use when parsing multiple files.

    Returns:
      metrics: A dictionary mapping string names to metric scores.
    """

    nq_gold_dict = util.read_annotation(gold_path, n_threads=num_threads)
    nq_pred_dict = util.read_prediction_json(prediction_path)
    long_answer_stats, short_answer_stats = score_answers(nq_gold_dict,
                                                          nq_pred_dict)

    return get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)


def get_metrics_with_answer_stats(long_answer_stats, short_answer_stats):
    """Generate metrics dict using long and short answer stats."""

    def _get_metric_dict(answer_stats, prefix=''):
        """Compute all metrics for a set of answer statistics."""
        opt_result, pr_table = compute_pr_curves(
            answer_stats, targets=[0.5, 0.75, 0.9])
        f1, precision, recall, threshold = opt_result
        metrics = OrderedDict({
            'best-threshold-f1': f1,
            'best-threshold-precision': precision,
            'best-threshold-recall': recall,
            'best-threshold': threshold,
        })
        for target, recall, precision, _ in pr_table:
            metrics['recall-at-precision>={:.2}'.format(target)] = recall
            metrics['precision-at-precision>={:.2}'.format(target)] = precision

        # Add prefix before returning.
        return dict([(prefix + k, v) for k, v in metrics.items()])

    metrics = _get_metric_dict(long_answer_stats, 'long-')
    metrics.update(_get_metric_dict(short_answer_stats, 'short-'))
    return metrics


def my_analysis(metrics, long_answer_stats, short_answer_stats):
    total_none_answer = 0
    total_none_correct = 0
    total_has_answer = 0
    total_has_correct = 0
    long_threshold = metrics["long-best-threshold"]
    for gold_has_answer, pred_has_answer, is_correct, score in long_answer_stats:
        if not gold_has_answer:
            total_none_answer += 1
            if score <= long_threshold:
                total_none_correct += 1

        if gold_has_answer:
            total_has_answer += 1
            if score >= long_threshold and is_correct:
                total_has_correct += 1

    metrics["long-none-accuracy"] = safe_divide(total_none_correct, total_none_answer)
    metrics["long-has-accuracy"] = safe_divide(total_has_correct, total_has_answer)

    # print(total_none_correct, total_none_answer)
    # print(total_has_correct, total_has_answer)

    total_none_answer = 0
    total_none_correct = 0
    total_has_answer = 0
    total_has_correct = 0
    short_threshold = metrics["long-best-threshold"]
    for gold_has_answer, pred_has_answer, is_correct, score in short_answer_stats:
        if not gold_has_answer:
            total_none_answer += 1
            if score <= short_threshold:
                total_none_correct += 1

        if gold_has_answer:
            total_has_answer += 1
            if score >= short_threshold and is_correct:
                total_has_correct += 1

    metrics["short-none-accuracy"] = safe_divide(total_none_correct, total_none_answer)
    metrics["short-has-accuracy"] = safe_divide(total_has_correct, total_has_answer)
    # print(total_none_correct, total_none_answer)
    # print(total_has_correct, total_has_answer)
    return metrics


def nq_evaluate(gold_path, predictions_path, num_threads):
    nq_gold_dict = util.read_annotation(
        gold_path, n_threads=num_threads)

    nq_pred_dict = util.read_prediction_json(predictions_path)

    long_answer_stats, short_answer_stats = score_answers(nq_gold_dict,
                                                          nq_pred_dict)

    metrics = get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)

    metrics = my_analysis(metrics, long_answer_stats, short_answer_stats)
    # scores = compute_final_f1(long_answer_stats, short_answer_stats)
    # return {"Long Answer Precision": scores['long-answer-precision'],
    #         "Long Answer Recall": scores['long-answer-recall'],
    #         "Long Answer F1": scores['long-answer-f1'],
    #         "Short Answer Precision": scores['short-answer-precision'],
    #         "Short Answer Recall": scores['short-answer-recall'],
    #         "Short Answer F1": scores['short-answer-f1']}
    return {"Long Answer Precision": metrics['long-best-threshold-precision'],
            "Long Answer Recall": metrics['long-best-threshold-recall'],
            "Long Answer F1": metrics['long-best-threshold-f1'],
            "Short Answer Precision": metrics['short-best-threshold-precision'],
            "Short Answer Recall": metrics['short-best-threshold-recall'],
            "Short Answer F1": metrics['short-best-threshold-f1'],
            "long None Accuracy:": metrics['long-none-accuracy'],
            "long Has Accuracy": metrics['long-has-accuracy'],
            "Short None Accuracy:": metrics['short-none-accuracy'],
            "Short Has Accuracy": metrics['short-has-accuracy']}
