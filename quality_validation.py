"""
quality_validation.py
---------------------
Language/script validation, quality scoring, and semantic analysis.
Provides hard rejection criteria for low-quality ASR/MT output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import config

log = config.get_logger(__name__)


# Script detection patterns for major languages
SCRIPT_PATTERNS = {
    "hi": {
        "script_name": "Devanagari",
        "range": r"[\u0900-\u097F]",
        "min_ratio": 0.3,  # At least 30% Devanagari characters
    },
    "bn": {
        "script_name": "Bengali",
        "range": r"[\u0980-\u09FF]",
        "min_ratio": 0.3,
    },
    "te": {
        "script_name": "Telugu",
        "range": r"[\u0C00-\u0C7F]",
        "min_ratio": 0.3,
    },
    "ta": {
        "script_name": "Tamil",
        "range": r"[\u0B80-\u0BFF]",
        "min_ratio": 0.3,
    },
    "mr": {
        "script_name": "Devanagari",
        "range": r"[\u0900-\u097F]",
        "min_ratio": 0.3,
    },
    "gu": {
        "script_name": "Gujarati",
        "range": r"[\u0A80-\u0AFF]",
        "min_ratio": 0.3,
    },
    "kn": {
        "script_name": "Kannada",
        "range": r"[\u0C80-\u0CFF]",
        "min_ratio": 0.3,
    },
    "ml": {
        "script_name": "Malayalam",
        "range": r"[\u0D00-\u0D7F]",
        "min_ratio": 0.3,
    },
    "pa": {
        "script_name": "Gurmukhi",
        "range": r"[\u0A00-\u0A7F]",
        "min_ratio": 0.3,
    },
    "ur": {
        "script_name": "Arabic",
        "range": r"[\u0600-\u06FF]",
        "min_ratio": 0.3,
    },
    "ar": {
        "script_name": "Arabic",
        "range": r"[\u0600-\u06FF]",
        "min_ratio": 0.3,
    },
    "zh": {
        "script_name": "CJK",
        "range": r"[\u4E00-\u9FFF]",
        "min_ratio": 0.3,
    },
    "ja": {
        "script_name": "CJK",
        "range": r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]",
        "min_ratio": 0.3,
    },
    "ko": {
        "script_name": "Hangul",
        "range": r"[\uAC00-\uD7AF]",
        "min_ratio": 0.3,
    },
    "en": {
        "script_name": "Latin",
        "range": r"[a-zA-Z]",
        "min_ratio": 0.5,
    },
    "es": {
        "script_name": "Latin",
        "range": r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]",
        "min_ratio": 0.5,
    },
    "fr": {
        "script_name": "Latin",
        "range": r"[a-zA-ZàâäæçéèêëïîôöœùûüÿÁÂÄÆÇÉÈÊËÏÎÔÖŒÙÛÜŸ]",
        "min_ratio": 0.5,
    },
    "de": {
        "script_name": "Latin",
        "range": r"[a-zA-ZäöüßÄÖÜ]",
        "min_ratio": 0.5,
    },
    "ru": {
        "script_name": "Cyrillic",
        "range": r"[\u0400-\u04FF]",
        "min_ratio": 0.4,
    },
}


@dataclass
class ValidationResult:
    """Result of quality validation."""
    passed: bool
    score: float  # 0.0 to 1.0
    reason: str
    details: dict


class TextQualityValidator:
    """Validates text quality for ASR and MT output."""
    
    # Quality thresholds
    MIN_SEGMENT_LENGTH = 5  # Minimum characters per segment
    MAX_REPETITION_RATIO = 0.25  # Max 25% repeated content
    MAX_GARBAGE_RATIO = 0.2  # Max 20% garbage characters
    MIN_SEMANTIC_SCORE = 0.3  # Minimum semantic coherence
    
    # Loop detection
    LOOP_MIN_REPEAT = 3  # Minimum repetitions to be considered a loop
    LOOP_MIN_LENGTH = 4  # Minimum phrase length for loop detection
    
    def __init__(self):
        self._common_loops = [
            r"(.)\1{4,}",  # Same character repeated 5+ times
            r"(\b\w+\b)(?:\s+\1){2,}",  # Same word repeated 3+ times
        ]
    
    def validate_translation(
        self,
        source_text: str,
        translated_text: str,
        src_lang: str,
        tgt_lang: str,
        strict: bool = True
    ) -> ValidationResult:
        """
        Comprehensive validation of translated text.
        
        Args:
            source_text: Original source text
            translated_text: Translated output
            src_lang: Source language code
            tgt_lang: Target language code
            strict: If True, apply hard rejection criteria
        
        Returns:
            ValidationResult with pass/fail status and score
        """
        details = {}
        scores = []
        failures = []
        
        # Check 1: Non-empty output
        if not translated_text or len(translated_text.strip()) < self.MIN_SEGMENT_LENGTH:
            return ValidationResult(
                passed=False,
                score=0.0,
                reason="Translation produced empty or too-short output",
                details={"translated_length": len(translated_text) if translated_text else 0}
            )
        
        # Check 2: Script validation
        script_result = self._validate_script(translated_text, tgt_lang)
        scores.append(script_result["score"])
        details["script_validation"] = script_result
        if strict and not script_result["valid"]:
            failures.append(f"Script validation failed: {script_result['reason']}")
        
        # Check 3: Loop/repetition detection
        loop_result = self._detect_loops(translated_text)
        scores.append(loop_result["score"])
        details["loop_detection"] = loop_result
        if strict and not loop_result["valid"]:
            failures.append(f"Loop detected: {loop_result['reason']}")
        
        # Check 4: Near-source copy detection
        copy_result = self._detect_near_source_copy(source_text, translated_text, src_lang, tgt_lang)
        scores.append(copy_result["score"])
        details["near_source_copy"] = copy_result
        if strict and not copy_result["valid"]:
            failures.append(f"Near-source copy detected: {copy_result['reason']}")
        
        # Check 5: Semantic coherence
        semantic_result = self._check_semantic_coherence(translated_text)
        scores.append(semantic_result["score"])
        details["semantic_coherence"] = semantic_result
        if strict and not semantic_result["valid"]:
            failures.append(f"Low semantic coherence: {semantic_result['reason']}")
        
        # Check 6: Garbage character ratio
        garbage_result = self._check_garbage_ratio(translated_text)
        scores.append(garbage_result["score"])
        details["garbage_ratio"] = garbage_result
        if strict and not garbage_result["valid"]:
            failures.append(f"High garbage ratio: {garbage_result['reason']}")
        
        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        if failures:
            return ValidationResult(
                passed=False,
                score=overall_score,
                reason="; ".join(failures),
                details=details
            )
        
        return ValidationResult(
            passed=True,
            score=overall_score,
            reason="All quality checks passed",
            details=details
        )
    
    def validate_asr_segment(
        self,
        text: str,
        expected_lang: Optional[str] = None,
        confidence: float = 0.0
    ) -> ValidationResult:
        """
        Validate ASR segment quality.
        
        Args:
            text: ASR transcript
            expected_lang: Expected language code
            confidence: ASR confidence score if available
        
        Returns:
            ValidationResult with pass/fail status
        """
        details = {}
        scores = []
        failures = []
        
        if not text or len(text.strip()) < 3:
            return ValidationResult(
                passed=False,
                score=0.0,
                reason="ASR produced empty or too-short segment",
                details={}
            )
        
        # Check repetition
        loop_result = self._detect_loops(text)
        scores.append(loop_result["score"])
        details["repetition"] = loop_result
        if not loop_result["valid"]:
            failures.append(f"Repetition detected: {loop_result['reason']}")
        
        # Check garbage ratio
        garbage_result = self._check_garbage_ratio(text)
        scores.append(garbage_result["score"])
        details["garbage"] = garbage_result
        if not garbage_result["valid"]:
            failures.append(f"High garbage ratio: {garbage_result['reason']}")
        
        # Check confidence if provided
        if confidence > 0 and confidence < 0.5:
            scores.append(confidence)
            failures.append(f"Low ASR confidence: {confidence:.2f}")
        
        # Check language if expected
        if expected_lang and expected_lang != "auto":
            script_result = self._validate_script(text, expected_lang)
            scores.append(script_result["score"])
            details["language"] = script_result
            if not script_result["valid"]:
                failures.append(f"Language mismatch: {script_result['reason']}")
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        if failures:
            return ValidationResult(
                passed=False,
                score=overall_score,
                reason="; ".join(failures),
                details=details
            )
        
        return ValidationResult(
            passed=True,
            score=overall_score,
            reason="ASR segment quality acceptable",
            details=details
        )
    
    def _validate_script(self, text: str, lang_code: str) -> dict:
        """Validate that text uses expected script."""
        if lang_code not in SCRIPT_PATTERNS:
            return {"valid": True, "score": 1.0, "reason": "No script validation for this language"}
        
        pattern = SCRIPT_PATTERNS[lang_code]
        script_chars = len(re.findall(pattern["range"], text))
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return {"valid": False, "score": 0.0, "reason": "No alphabetic characters found"}
        
        ratio = script_chars / total_chars
        min_ratio = pattern["min_ratio"]
        
        # Score based on how close to expected ratio
        score = min(1.0, ratio / min_ratio) if ratio < min_ratio else 1.0
        
        if ratio < min_ratio * 0.5:
            return {
                "valid": False,
                "score": score,
                "reason": f"Script mismatch: expected {pattern['script_name']} ({ratio:.2%} vs {min_ratio:.0%} min)",
                "detected_ratio": ratio,
                "expected_script": pattern["script_name"]
            }
        
        return {
            "valid": ratio >= min_ratio * 0.8,
            "score": score,
            "reason": f"Script validation passed ({ratio:.1%} {pattern['script_name']})",
            "detected_ratio": ratio
        }
    
    def _detect_loops(self, text: str) -> dict:
        """Detect repetitive/looped content."""
        if not text:
            return {"valid": True, "score": 1.0, "reason": "Empty text"}
        
        words = text.split()
        if len(words) < self.LOOP_MIN_LENGTH:
            return {"valid": True, "score": 1.0, "reason": "Text too short for loop detection"}
        
        # Check for character repetition
        for pattern in self._common_loops:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return {
                    "valid": False,
                    "score": 0.0,
                    "reason": f"Repetition pattern detected: {matches[0] if matches else 'unknown'}",
                    "pattern": pattern
                }
        
        # Check for phrase repetition (2-5 word phrases)
        for phrase_len in range(5, 1, -1):
            if len(words) < phrase_len * 2:
                continue
            
            phrases = []
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i + phrase_len]).lower()
                phrases.append(phrase)
            
            # Count phrase frequencies
            from collections import Counter
            phrase_counts = Counter(phrases)
            
            for phrase, count in phrase_counts.most_common(3):
                if count >= self.LOOP_MIN_REPEAT:
                    return {
                        "valid": False,
                        "score": 0.1,
                        "reason": f"Phrase repeated {count} times: '{phrase[:50]}...'" if len(phrase) > 50 else f"Phrase repeated {count} times: '{phrase}'",
                        "repeated_phrase": phrase,
                        "repeat_count": count
                    }
        
        # Calculate repetition score
        unique_words = len(set(w.lower() for w in words))
        total_words = len(words)
        diversity = unique_words / total_words if total_words > 0 else 0
        
        return {
            "valid": True,
            "score": diversity,
            "reason": f"No loops detected (diversity: {diversity:.2f})",
            "diversity_score": diversity
        }
    
    def _detect_near_source_copy(
        self,
        source: str,
        translated: str,
        src_lang: str,
        tgt_lang: str
    ) -> dict:
        """Detect if translation is just a near-copy of source."""
        if src_lang == tgt_lang:
            return {"valid": True, "score": 1.0, "reason": "Same language, copy expected"}
        
        # Normalize both texts
        src_norm = re.sub(r"[^\w]", "", source.lower())
        tgt_norm = re.sub(r"[^\w]", "", translated.lower())
        
        if not src_norm or not tgt_norm:
            return {"valid": True, "score": 1.0, "reason": "Cannot compare empty texts"}
        
        # Calculate similarity
        import difflib
        similarity = difflib.SequenceMatcher(None, src_norm, tgt_norm).ratio()
        
        # For different scripts, similarity should be very low
        src_script = SCRIPT_PATTERNS.get(src_lang, {}).get("script_name", "Unknown")
        tgt_script = SCRIPT_PATTERNS.get(tgt_lang, {}).get("script_name", "Unknown")
        
        if src_script != tgt_script:
            # Different scripts - similarity should be < 0.3
            threshold = 0.3
        else:
            # Same script family - allow higher similarity
            threshold = 0.5
        
        if similarity > threshold:
            return {
                "valid": False,
                "score": max(0, 1 - similarity),
                "reason": f"Near-source copy detected (similarity: {similarity:.2%})",
                "similarity": similarity,
                "threshold": threshold
            }
        
        return {
            "valid": True,
            "score": max(0, 1 - similarity),
            "reason": f"Translation differs from source (similarity: {similarity:.1%})",
            "similarity": similarity
        }
    
    def _check_semantic_coherence(self, text: str) -> dict:
        """Check if text has semantic coherence (basic heuristics)."""
        if not text:
            return {"valid": False, "score": 0.0, "reason": "Empty text"}
        
        sentences = [s.strip() for s in re.split(r'[.!?।॥]+', text) if s.strip()]
        
        if not sentences:
            return {"valid": False, "score": 0.0, "reason": "No valid sentences found"}
        
        # Check sentence length distribution
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        # Very short or very long sentences are suspicious
        if avg_length < 2:
            return {
                "valid": False,
                "score": 0.2,
                "reason": f"Suspiciously short sentences (avg {avg_length:.1f} words)",
                "avg_sentence_length": avg_length
            }
        
        if avg_length > 50:
            return {
                "valid": False,
                "score": 0.3,
                "reason": f"Suspiciously long sentences (avg {avg_length:.1f} words)",
                "avg_sentence_length": avg_length
            }
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 3:
            return {
                "valid": False,
                "score": 0.1,
                "reason": f"Too few words ({len(words)})",
                "word_count": len(words)
            }
        
        # Score based on sentence quality
        good_sentences = sum(1 for l in lengths if 3 <= l <= 30)
        score = good_sentences / len(sentences) if sentences else 0
        
        return {
            "valid": score >= self.MIN_SEMANTIC_SCORE,
            "score": score,
            "reason": f"Semantic coherence: {score:.2f} ({good_sentences}/{len(sentences)} good sentences)",
            "sentence_count": len(sentences),
            "avg_length": avg_length
        }
    
    def _check_garbage_ratio(self, text: str) -> dict:
        """Check ratio of garbage/non-linguistic characters."""
        if not text:
            return {"valid": True, "score": 1.0, "reason": "Empty text"}
        
        # Define garbage patterns
        garbage_patterns = [
            r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
            r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
            r"\u0D00-\u0D7F\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF"
            r"\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF.,!?;:'\"-]",
        ]
        
        garbage_chars = 0
        for pattern in garbage_patterns:
            garbage_chars += len(re.findall(pattern, text))
        
        total_chars = len(text)
        ratio = garbage_chars / total_chars if total_chars > 0 else 0
        
        score = max(0, 1 - (ratio / self.MAX_GARBAGE_RATIO))
        
        if ratio > self.MAX_GARBAGE_RATIO:
            return {
                "valid": False,
                "score": score,
                "reason": f"High garbage ratio: {ratio:.1%}",
                "garbage_ratio": ratio
            }
        
        return {
            "valid": True,
            "score": score,
            "reason": f"Garbage ratio acceptable: {ratio:.1%}",
            "garbage_ratio": ratio
        }


class TimestampValidator:
    """Validates and corrects ASR segment timestamps."""
    
    # Timing constraints
    MIN_SEGMENT_DURATION = 0.5  # Minimum 500ms per segment
    MAX_SEGMENT_DURATION = 15.0  # Maximum 15 seconds per segment
    MAX_GAP_BETWEEN_SEGMENTS = 3.0  # Max 3 second gap
    MIN_WORDS_PER_SECOND = 0.5  # Minimum speech rate
    MAX_WORDS_PER_SECOND = 8.0  # Maximum speech rate (rushed)
    
    def validate_and_fix_segments(self, segments: list[dict]) -> list[dict]:
        """
        Validate and fix segment timestamps.
        
        Args:
            segments: List of segments with 'start', 'end', 'text' keys
        
        Returns:
            List of validated and potentially fixed segments
        """
        if not segments:
            return []
        
        validated = []
        
        for i, seg in enumerate(segments):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            text = seg.get("text", "")
            
            # Fix inverted timestamps
            if end < start:
                log.warning("Segment %d: inverted timestamps (%.2f > %.2f), swapping", i, start, end)
                start, end = end, start
            
            # Fix zero or negative duration
            duration = end - start
            if duration < self.MIN_SEGMENT_DURATION:
                # Try to extend based on word count
                words = len(text.split())
                estimated_duration = max(
                    self.MIN_SEGMENT_DURATION,
                    words / self.MAX_WORDS_PER_SECOND
                )
                end = start + estimated_duration
                log.warning("Segment %d: duration too short (%.2fs), extended to %.2fs", 
                          i, duration, estimated_duration)
                duration = estimated_duration
            
            if duration > self.MAX_SEGMENT_DURATION:
                # Split long segments
                log.warning("Segment %d: duration too long (%.2fs), may need splitting", 
                          i, duration)
                # Keep as-is but mark for potential splitting
            
            # Check speech rate
            words = len(text.split())
            words_per_sec = words / duration if duration > 0 else 0
            
            if words_per_sec > self.MAX_WORDS_PER_SECOND:
                log.warning("Segment %d: rushed speech (%.1f words/sec), extending duration",
                          i, words_per_sec)
                # Extend duration to match reasonable speech rate
                end = start + (words / 4.0)  # Assume 4 words/sec
            
            if words_per_sec < self.MIN_WORDS_PER_SECOND and len(text) > 10:
                log.warning("Segment %d: very slow speech (%.1f words/sec), possible silence/music",
                          i, words_per_sec)
            
            # Check gap from previous segment
            if validated:
                prev_end = validated[-1]["end"]
                gap = start - prev_end
                if gap > self.MAX_GAP_BETWEEN_SEGMENTS:
                    log.warning("Segment %d: large gap from previous (%.2fs)", i, gap)
            
            validated.append({
                "start": start,
                "end": end,
                "text": text,
                "duration": end - start,
                "words_per_second": words_per_sec,
            })
        
        return validated
    
    def detect_music_or_noise(self, segments: list[dict]) -> list[int]:
        """
        Detect segments that are likely music or noise.
        
        Returns:
            List of segment indices that appear to be music/noise
        """
        suspicious_indices = []
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            duration = seg.get("end", 0) - seg.get("start", 0)
            words = len(text.split())
            
            # Low word density = likely music/noise
            if duration > 2.0 and words / duration < 0.3:
                suspicious_indices.append(i)
                continue
            
            # Check for music-related patterns
            music_patterns = [
                r"\b(music|instrumental|bgm|background)\b",
                r"[♪♫]",
                r"(\w+)\s+\1\s+\1",  # Triple repetition
            ]
            for pattern in music_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    suspicious_indices.append(i)
                    break
        
        return suspicious_indices


# Global validator instance
_validator: Optional[TextQualityValidator] = None
_timestamp_validator: Optional[TimestampValidator] = None


def get_validator() -> TextQualityValidator:
    """Get singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = TextQualityValidator()
    return _validator


def get_timestamp_validator() -> TimestampValidator:
    """Get singleton timestamp validator instance."""
    global _timestamp_validator
    if _timestamp_validator is None:
        _timestamp_validator = TimestampValidator()
    return _timestamp_validator


def validate_translation(
    source: str,
    translated: str,
    src_lang: str,
    tgt_lang: str,
    strict: bool = True
) -> ValidationResult:
    """Convenience function for translation validation."""
    return get_validator().validate_translation(source, translated, src_lang, tgt_lang, strict)


def validate_asr_segment(
    text: str,
    expected_lang: Optional[str] = None,
    confidence: float = 0.0
) -> ValidationResult:
    """Convenience function for ASR validation."""
    return get_validator().validate_asr_segment(text, expected_lang, confidence)


def validate_timestamps(segments: list[dict]) -> list[dict]:
    """Convenience function for timestamp validation."""
    return get_timestamp_validator().validate_and_fix_segments(segments)


def detect_music_noise(segments: list[dict]) -> list[int]:
    """Convenience function for music/noise detection."""
    return get_timestamp_validator().detect_music_or_noise(segments)
