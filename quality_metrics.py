"""
quality_metrics.py
------------------
Evaluation framework for tracking translation quality metrics.
Supports WER, chrF, BLEU, and human MOS scoring.
Provides benchmark-based regression testing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import config

log = config.get_logger(__name__)


@dataclass
class TranslationMetrics:
    """Metrics for a single translation evaluation."""
    source: str
    reference: str  # Human reference translation
    hypothesis: str  # System output
    src_lang: str
    tgt_lang: str
    
    # Automatic metrics
    wer: Optional[float] = None  # Word Error Rate (lower is better)
    chrf: Optional[float] = None  # chrF score (higher is better, 0-100)
    bleu: Optional[float] = None  # BLEU score (higher is better, 0-100)
    
    # Custom metrics
    repetition_score: Optional[float] = None
    script_match_score: Optional[float] = None
    semantic_score: Optional[float] = None
    
    # Human evaluation (MOS)
    mos_quality: Optional[float] = None  # 1-5 scale
    mos_naturalness: Optional[float] = None  # 1-5 scale
    
    # Metadata
    timestamp: str = ""
    backend: str = ""
    segment_index: Optional[int] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MetricsCalculator:
    """Calculate various translation quality metrics."""
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate using Levenshtein distance.
        WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                      dp[i][j-1],       # insertion
                                      dp[i-1][j-1])     # substitution
        
        # WER = edit distance / reference length
        wer = dp[m][n] / m if m > 0 else 0.0
        return min(wer, 1.0)  # Cap at 100%
    
    @staticmethod
    def calculate_chrf(reference: str, hypothesis: str, ngram_order: int = 6) -> float:
        """
        Calculate chrF (character n-gram F-score).
        Simplified implementation - for production use sacrebleu.
        """
        def get_ngrams(text: str, n: int) -> dict:
            ngrams = {}
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
            return ngrams
        
        # Calculate for each n-gram order
        precisions = []
        recalls = []
        
        for n in range(1, ngram_order + 1):
            ref_ngrams = get_ngrams(reference, n)
            hyp_ngrams = get_ngrams(hypothesis, n)
            
            if not hyp_ngrams:
                continue
            
            # Count matches
            matches = 0
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            precision = matches / sum(hyp_ngrams.values()) if hyp_ngrams else 0
            recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Average precision and recall
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        
        # F-score with beta=3 (recall weighted)
        beta = 3
        if avg_precision + avg_recall == 0:
            return 0.0
        
        chrf = (1 + beta**2) * avg_precision * avg_recall / (beta**2 * avg_precision + avg_recall)
        return chrf * 100  # Return as percentage
    
    @staticmethod
    def calculate_bleu(reference: str, hypothesis: str, max_ngram: int = 4) -> float:
        """
        Calculate BLEU score.
        Simplified implementation - for production use sacrebleu.
        """
        def get_ngrams(text: str, n: int) -> dict:
            words = text.split()
            ngrams = {}
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
            return ngrams
        
        # Brevity penalty
        ref_len = len(reference.split())
        hyp_len = len(hypothesis.split())
        
        if hyp_len == 0:
            return 0.0
        
        bp = 1.0 if hyp_len > ref_len else (0.0 if ref_len == 0 else (1 - ref_len/hyp_len))
        
        # N-gram precisions
        geo_mean = 1.0
        valid_n = 0
        
        for n in range(1, max_ngram + 1):
            ref_ngrams = get_ngrams(reference, n)
            hyp_ngrams = get_ngrams(hypothesis, n)
            
            if not hyp_ngrams:
                continue
            
            matches = sum(min(count, ref_ngrams.get(ngram, 0)) 
                         for ngram, count in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            
            if total > 0:
                precision = matches / total
                if precision > 0:
                    geo_mean *= precision
                    valid_n += 1
        
        if valid_n == 0:
            return 0.0
        
        geo_mean **= (1 / valid_n)
        bleu = bp * geo_mean
        
        return bleu * 100  # Return as percentage
    
    @staticmethod
    def calculate_repetition_score(text: str) -> float:
        """Calculate repetition penalty score (0-1, higher is better)."""
        words = text.split()
        if len(words) < 5:
            return 1.0
        
        unique_words = len(set(w.lower() for w in words))
        total_words = len(words)
        
        return unique_words / total_words
    
    @classmethod
    def calculate_all(
        cls,
        reference: str,
        hypothesis: str,
        src_lang: str,
        tgt_lang: str
    ) -> dict:
        """Calculate all available metrics."""
        return {
            "wer": cls.calculate_wer(reference, hypothesis),
            "chrf": cls.calculate_chrf(reference, hypothesis),
            "bleu": cls.calculate_bleu(reference, hypothesis),
            "repetition_score": cls.calculate_repetition_score(hypothesis),
        }


class BenchmarkManager:
    """
    Manage benchmark datasets for regression testing.
    
    A benchmark is a JSON file containing:
    {
        "name": "hindi_automotive_v1",
        "description": "Hindi automotive content test set",
        "src_lang": "en",
        "tgt_lang": "hi",
        "segments": [
            {
                "source": "English text",
                "reference": "Human translation",
                "context": "optional domain info"
            }
        ]
    }
    """
    
    def __init__(self, benchmark_dir: Optional[str | Path] = None):
        self.benchmark_dir = Path(benchmark_dir or config.BENCHMARK_DIR)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = MetricsCalculator()
    
    def load_benchmark(self, name: str) -> dict:
        """Load a benchmark by name."""
        path = self.benchmark_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    
    def save_benchmark(self, name: str, data: dict) -> None:
        """Save a benchmark."""
        path = self.benchmark_dir / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Saved benchmark: %s", path)
    
    def list_benchmarks(self) -> list[str]:
        """List available benchmarks."""
        return [p.stem for p in self.benchmark_dir.glob("*.json")]
    
    def run_evaluation(
        self,
        benchmark_name: str,
        translate_func: Callable[[str, str, str], str],
        backend_name: str = "unknown"
    ) -> dict:
        """
        Run evaluation on a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to run
            translate_func: Function(source, src_lang, tgt_lang) -> translation
            backend_name: Name of the translation backend
        
        Returns:
            Evaluation results with aggregated metrics
        """
        benchmark = self.load_benchmark(benchmark_name)
        segments = benchmark.get("segments", [])
        
        if not segments:
            raise ValueError(f"Benchmark {benchmark_name} has no segments")
        
        src_lang = benchmark.get("src_lang", "en")
        tgt_lang = benchmark.get("tgt_lang", "hi")
        
        results = []
        all_metrics = []
        
        log.info("Running benchmark '%s' with %d segments...", benchmark_name, len(segments))
        
        for i, segment in enumerate(segments):
            source = segment.get("source", "")
            reference = segment.get("reference", "")
            
            if not source or not reference:
                log.warning("Skipping segment %d: missing source or reference", i)
                continue
            
            # Run translation
            try:
                hypothesis = translate_func(source, src_lang, tgt_lang)
            except Exception as e:
                log.error("Translation failed for segment %d: %s", i, e)
                continue
            
            # Calculate metrics
            metrics = TranslationMetrics(
                source=source,
                reference=reference,
                hypothesis=hypothesis,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                backend=backend_name,
                segment_index=i,
            )
            
            # Calculate automatic metrics
            calculated = self.calculator.calculate_all(reference, hypothesis, src_lang, tgt_lang)
            metrics.wer = calculated["wer"]
            metrics.chrf = calculated["chrf"]
            metrics.bleu = calculated["bleu"]
            metrics.repetition_score = calculated["repetition_score"]
            
            results.append(metrics)
            all_metrics.append({
                "wer": metrics.wer,
                "chrf": metrics.chrf,
                "bleu": metrics.bleu,
            })
            
            log.debug("Segment %d: WER=%.3f chrF=%.2f BLEU=%.2f",
                     i, metrics.wer, metrics.chrf, metrics.bleu)
        
        # Aggregate results
        if not all_metrics:
            raise RuntimeError("No valid results from benchmark")
        
        aggregated = {
            "benchmark_name": benchmark_name,
            "backend": backend_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "num_segments": len(results),
            "timestamp": datetime.now().isoformat(),
            "aggregated": {
                "wer_mean": sum(m["wer"] for m in all_metrics) / len(all_metrics),
                "wer_median": sorted(m["wer"] for m in all_metrics)[len(all_metrics)//2],
                "chrf_mean": sum(m["chrf"] for m in all_metrics) / len(all_metrics),
                "bleu_mean": sum(m["bleu"] for m in all_metrics) / len(all_metrics),
            },
            "segments": [asdict(r) for r in results],
        }
        
        # Save results
        if config.BENCHMARK_AUTO_SAVE:
            result_path = self.benchmark_dir / f"{benchmark_name}_{backend_name}_results.json"
            result_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), 
                                  encoding="utf-8")
            log.info("Saved results to %s", result_path)
        
        return aggregated
    
    def compare_results(self, result1: dict, result2: dict) -> dict:
        """
        Compare two evaluation results.
        
        Returns:
            Comparison with deltas and significance
        """
        agg1 = result1.get("aggregated", {})
        agg2 = result2.get("aggregated", {})
        
        comparison = {
            "result1": {
                "benchmark": result1.get("benchmark_name"),
                "backend": result1.get("backend"),
                "timestamp": result1.get("timestamp"),
            },
            "result2": {
                "benchmark": result2.get("benchmark_name"),
                "backend": result2.get("backend"),
                "timestamp": result2.get("timestamp"),
            },
            "deltas": {},
            "improvements": {},
        }
        
        # Compare metrics
        for metric in ["wer_mean", "chrf_mean", "bleu_mean"]:
            v1 = agg1.get(metric, 0)
            v2 = agg2.get(metric, 0)
            delta = v2 - v1
            
            # For WER, lower is better; for chrF/BLEU, higher is better
            is_better = (metric == "wer_mean" and delta < 0) or \
                       (metric != "wer_mean" and delta > 0)
            
            comparison["deltas"][metric] = {
                "result1": v1,
                "result2": v2,
                "delta": delta,
                "delta_percent": (delta / v1 * 100) if v1 != 0 else 0,
            }
            comparison["improvements"][metric] = is_better
        
        return comparison


class RegressionTracker:
    """
    Track metrics over time to detect regressions.
    """
    
    def __init__(self, history_file: Optional[str | Path] = None):
        self.history_file = Path(history_file or config.BENCHMARK_DIR) / "metric_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> dict:
        """Load metric history from file."""
        if self.history_file.exists():
            return json.loads(self.history_file.read_text(encoding="utf-8"))
        return {}
    
    def _save_history(self) -> None:
        """Save metric history to file."""
        self.history_file.write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    
    def record(self, benchmark_name: str, backend: str, metrics: dict) -> None:
        """Record a new metric entry."""
        key = f"{benchmark_name}_{backend}"
        
        if key not in self.history:
            self.history[key] = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        
        self.history[key].append(entry)
        self._save_history()
        
        log.info("Recorded metrics for %s/%s", benchmark_name, backend)
    
    def check_regression(
        self,
        benchmark_name: str,
        backend: str,
        current_metrics: dict,
        threshold_percent: float = 5.0
    ) -> dict:
        """
        Check if current metrics show regression vs. history.
        
        Returns:
            Regression report with alerts
        """
        key = f"{benchmark_name}_{backend}"
        history = self.history.get(key, [])
        
        if len(history) < 2:
            return {"status": "insufficient_history", "alerts": []}
        
        # Get previous best
        prev_best = max(history[:-1], 
                       key=lambda x: x["metrics"].get("chrf_mean", 0) - x["metrics"].get("wer_mean", 0))
        prev_metrics = prev_best["metrics"]
        
        alerts = []
        
        # Check for regressions
        for metric, current in current_metrics.items():
            if metric not in prev_metrics:
                continue
            
            previous = prev_metrics[metric]
            
            # For WER, increase is bad; for others, decrease is bad
            if metric == "wer_mean":
                delta = current - previous
                is_regression = delta > 0
            else:
                delta = previous - current
                is_regression = delta > 0
            
            if is_regression:
                delta_percent = abs(delta / previous * 100) if previous != 0 else 0
                
                if delta_percent > threshold_percent:
                    alerts.append({
                        "metric": metric,
                        "previous": previous,
                        "current": current,
                        "delta": delta,
                        "delta_percent": delta_percent,
                        "severity": "critical" if delta_percent > 10 else "warning",
                    })
        
        return {
            "status": "regression_detected" if alerts else "ok",
            "alerts": alerts,
            "previous_best": prev_metrics,
            "current": current_metrics,
        }


# Global instances
_metrics_calculator: Optional[MetricsCalculator] = None
_benchmark_manager: Optional[BenchmarkManager] = None
_regression_tracker: Optional[RegressionTracker] = None


def get_metrics_calculator() -> MetricsCalculator:
    """Get singleton metrics calculator."""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = MetricsCalculator()
    return _metrics_calculator


def get_benchmark_manager() -> BenchmarkManager:
    """Get singleton benchmark manager."""
    global _benchmark_manager
    if _benchmark_manager is None:
        _benchmark_manager = BenchmarkManager()
    return _benchmark_manager


def get_regression_tracker() -> RegressionTracker:
    """Get singleton regression tracker."""
    global _regression_tracker
    if _regression_tracker is None:
        _regression_tracker = RegressionTracker()
    return _regression_tracker


def evaluate_translation(
    reference: str,
    hypothesis: str,
    src_lang: str = "en",
    tgt_lang: str = "hi"
) -> dict:
    """Convenience function to evaluate a single translation."""
    return get_metrics_calculator().calculate_all(reference, hypothesis, src_lang, tgt_lang)


def run_benchmark(
    benchmark_name: str,
    translate_func: Callable[[str, str, str], str],
    backend_name: str = "unknown"
) -> dict:
    """Convenience function to run a benchmark."""
    return get_benchmark_manager().run_evaluation(benchmark_name, translate_func, backend_name)
