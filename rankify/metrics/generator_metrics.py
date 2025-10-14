# rankify/metrics/generator_metrics.py
from typing import Dict, List, Optional
import warnings

from .base_metrics import ExactMatch, F1Score, PrecisionScore, RecallScore, ContainsMatch, BLEUScore
from .ragas_bridge import RagasModels, build_ragas_dataset

class GeneratorMetrics:
    """
    Generation-side metrics:
      - Classic: EM, F1, Precision, Recall, Contains, BLEU (yours)
      - ROUGE-L (native impl or via ragas RougeScore)
      - BERTScore (bert-score package)
      - RAGAS: Response/Answer Relevancy, Faithfulness
    """

    def __init__(self, documents, config: Optional[Dict]=None):
        self.documents = documents
        self.config = config or {"dataset_name": "QA_Evaluation"}

    # ---------- classic ----------
    def classic(self, predictions: List[str]) -> Dict[str, float]:
        data = type("Data", (object,), {"documents": self.documents, "predictions": predictions})()
        metrics = [ExactMatch(self.config), F1Score(self.config), PrecisionScore(self.config),
                   RecallScore(self.config), ContainsMatch(self.config), BLEUScore(self.config)]
        out = {}
        for m in metrics:
            score, _ = m.calculate_metric(data)
            out.update(score)
        return out

    # ---------- ROUGE-L ----------
    def rouge_l(self, predictions: List[str]) -> Dict[str, float]:
        try:
            # lightweight pure python via ragas' RougeScore if available
            from ragas.dataset_schema import SingleTurnSample, RagasDataset
            from ragas.metrics import RougeScore
            samples = []
            for doc, pred in zip(self.documents, predictions):
                # reference must be a string; join if multiple
                ref = " ".join(getattr(doc.answers, "answers", getattr(doc, "answers", [])) or [])
                samples.append(SingleTurnSample(response=pred or "", reference=ref))
            ds = RagasDataset.from_list(samples)
            rouge = RougeScore(rouge_type="rougeL", mode="fmeasure")
            from ragas import evaluate
            res = evaluate(ds, metrics=[rouge])
            # average per-sample score
            return {"rougeL_f1": float(res["rouge_score"])}
        except Exception:
            # fallback: simple LCS-based ROUGE-L
            def lcs(a, b):
                m, n = len(a), len(b)
                dp = [[0]*(n+1) for _ in range(m+1)]
                for i in range(m):
                    for j in range(n):
                        dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
                return dp[m][n]
            def score(pred, refs):
                pred_tokens = (pred or "").split()
                best = 0.0
                for r in refs:
                    ref_tokens = (r or "").split()
                    l = lcs(pred_tokens, ref_tokens)
                    prec = l/len(pred_tokens) if pred_tokens else 0
                    rec  = l/len(ref_tokens)  if ref_tokens else 0
                    f1 = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0
                    best = max(best, f1)
                return best
            refs = []
            for doc in self.documents:
                ans = getattr(doc.answers, "answers", getattr(doc, "answers", []))
                refs.append(ans if isinstance(ans, list) else [str(ans)])
            vals = [score(p, r) for p, r in zip(predictions, refs)]
            return {"rougeL_f1": sum(vals)/len(vals) if vals else 0.0}

    # ---------- BERTScore ----------
    def bertscore(self, predictions: List[str]) -> Dict[str, float]:
        try:
            # pip install bert-score
            from bert_score import score as bert_score
            # choose a single best reference per item (max F1 against all refs)
            refs_list = []
            for doc in self.documents:
                ans = getattr(doc.answers, "answers", getattr(doc, "answers", []))
                refs_list.append(ans if isinstance(ans, list) else [str(ans)])

            # naive: take first ref; advanced: pick max per row
            cands = []
            refs  = []
            for p, refs_row in zip(predictions, refs_list):
                if not refs_row:
                    refs_row = [""]
                # pick the longest reference (proxy for informativeness)
                best_ref = max(refs_row, key=lambda s: len(s))
                cands.append(p or "")
                refs.append(best_ref)

            P, R, F1 = bert_score(cands, refs, lang="en", rescale_with_baseline=True)
            return {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
                "bertscore_f1": float(F1.mean()),
            }
        except Exception as e:
            warnings.warn(f"BERTScore unavailable (install bert-score): {e}")
            return {"bertscore_f1": 0.0}

    # ---------- RAGAS (generator) ----------
    def ragas_generator(self, predictions: List[str], judge: RagasModels, use_reordered_contexts: bool = True) -> Dict[str, float]:
        """
        Uses RAGAS ResponseRelevancy + Faithfulness.
        """
        try:
            from ragas.metrics import ResponseRelevancy, Faithfulness
            from ragas import evaluate
            dataset = build_ragas_dataset(self.documents, predictions, use_reordered=use_reordered_contexts)
            llm, embeddings = judge.build()
            metrics = [ResponseRelevancy(llm=llm, embeddings=embeddings), Faithfulness(llm=llm)]
            result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
            # result is a dict-like; keys follow metric names
            return {
                "ragas_response_relevancy": float(result["response_relevancy"]),
                "ragas_faithfulness": float(result["faithfulness"]),
            }
        except Exception as e:
            warnings.warn(f"RAGAS generator metrics failed: {e}")
            return {"ragas_response_relevancy": 0.0, "ragas_faithfulness": 0.0}

    # ---------- one-call convenience ----------
    def all(self, predictions: List[str], ragas_models: Optional[RagasModels] = None, use_reordered_contexts: bool = True) -> Dict[str, float]:
        out = {}
        out.update(self.classic(predictions))
        out.update(self.rouge_l(predictions))
        out.update(self.bertscore(predictions))
        if ragas_models:
            out.update(self.ragas_generator(predictions, judge=ragas_models, use_reordered_contexts=use_reordered_contexts))
        return out
