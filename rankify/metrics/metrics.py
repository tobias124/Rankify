import re
import numpy as np
import collections
import math
from functools import lru_cache
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return re.sub(r'[^a-z0-9 ]', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(s.lower())))


class BaseMetric:
    """Base class for evaluation metrics."""
    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.get("dataset_name", "default")

    def calculate_metric(self, data):
        return {}, []

    def get_dataset_answer(self, data):
        return [doc.answers.answers for doc in data.documents]


### **Generation Metrics (Exact Match, F1, Precision, Recall, BLEU)**

class ExactMatch(BaseMetric):
    """Exact Match Score"""
    metric_name = "exact_match"

    def calculate_em(self, prediction, golden_answers):
        normalized_prediction = normalize_answer(prediction)
        for answer in golden_answers:
            if normalize_answer(answer) == normalized_prediction:
                return 1.0
        return 0.0

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"exact_match": score *100 }, metric_score_list


class F1Score(BaseMetric):
    """F1 Score Calculation"""
    metric_name = "f1_score"

    def token_level_scores(self, prediction, ground_truths):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        for ground_truth in ground_truths:
            pred_tokens = normalize_answer(prediction).split()
            truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

            final_metric["f1"] = max(f1, final_metric["f1"])
            final_metric["precision"] = max(precision, final_metric["precision"])
            final_metric["recall"] = max(recall, final_metric["recall"])

        return final_metric

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"f1_score": score*100 }, metric_score_list


class PrecisionScore(F1Score):
    """Precision Score Calculation"""
    metric_name = "precision"

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"precision": score*100}, metric_score_list


class RecallScore(F1Score):
    """Recall Score Calculation"""
    metric_name = "recall"

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"recall": score*100}, metric_score_list

class ContainsMatch(BaseMetric):
    """Checks if any reference answer is contained within the predicted answer."""
    metric_name = "contains_match"

    def calculate_contains(self, prediction, golden_answers):
        normalized_prediction = normalize_answer(prediction)
        for answer in golden_answers:
            if normalize_answer(answer) in normalized_prediction:
                return 1.0  # The prediction contains the answer
        return 0.0

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_contains(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"contains_match": score*100}, metric_score_list
    

class BLEUScore(BaseMetric):
    """BLEU Score Calculation"""
    metric_name = "bleu_score"

    def __init__(self, config):
        super().__init__(config)
        self.max_order = config.get("bleu_max_order", 4)
        self.smooth = config.get("bleu_smooth", False)

    def compute_bleu(self, reference_corpus, translation_corpus):
        matches_by_order = [0] * self.max_order
        possible_matches_by_order = [0] * self.max_order
        reference_length = 0
        translation_length = 0

        for references, translation in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, self.max_order)
            translation_ngram_counts = self._get_ngrams(translation, self.max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, self.max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * self.max_order
        for i in range(self.max_order):
            if self.smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
            else:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i] if possible_matches_by_order[i] > 0 else 0.0

        bleu = math.exp(sum(math.log(p) for p in precisions) / self.max_order) if min(precisions) > 0 else 0.0
        return bleu
    
    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.

        Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.

        Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i : i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def calculate_metric(self, data):
        pred_list = [normalize_answer(pred).split() for pred in data.predictions]
        golden_answers_list = [[normalize_answer(ans).split() for ans in ans_list] for ans_list in self.get_dataset_answer(data)]

        bleu_score = self.compute_bleu(golden_answers_list, pred_list)
        return {"bleu_score": bleu_score}, [bleu_score] * len(pred_list)


### **Retrieval Metrics (Top-K Accuracy)**

class Metrics:
    """Main class to calculate retrieval and generation metrics."""

    def __init__(self, documents):
        self.documents = documents
        #self.predictions = predictions
        self.config = {"dataset_name": "QA_Evaluation"}

    def top_k_accuracy(self, k, use_reordered=False):
        """Calculates top-k retrieval accuracy."""
        hits, total = 0, 0
        for document in self.documents:
            contexts_to_use = document.reorder_contexts if (use_reordered and document.reorder_contexts) else document.contexts
            if any(context.has_answer for context in contexts_to_use[:k]):
                hits += 1
            total += 1
        return (hits / total) * 100 if total > 0 else 0

    def calculate_retrieval_metrics(self, ks=[1, 5, 10, 20, 50, 100], use_reordered=False):
        """Calculates top-k retrieval metrics."""
        results = {}
        for k in ks:
            results[f'top_{k}'] = self.top_k_accuracy(k, use_reordered)
        return results

    def calculate_generation_metrics(self, predictions):

        """Calculates generation metrics."""
        data = type("Data", (object,), {"documents": self.documents, "predictions": predictions})()
        metric_classes = [ExactMatch, F1Score, PrecisionScore, RecallScore, ContainsMatch]

        results = {}
        for metric_class in metric_classes:
            metric = metric_class(self.config)
            score, _ = metric.calculate_metric(data)
            results.update(score)

        return results
