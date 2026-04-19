"""
translate.py
------------
Modular translation layer with swappable backends.
FIXED: Smart backend routing, semantic chunking, quality gates, glossary support.
"""

from __future__ import annotations

import gc
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import config
from quality_validation import validate_translation, ValidationResult
from glossary import lock_terms, restore_terms, verify_terms, get_glossary

log = config.get_logger(__name__)

# 8GB OPTIMIZATION: Smaller chunks to prevent GPU OOM
MAX_CHUNK_CHARS = 1200  # Reduced from 2000 for 8GB cards
MAX_GPU_CACHE = 2

# Semantic chunking parameters
SEMANTIC_CHUNK_MIN_CHARS = 100
SEMANTIC_CHUNK_MAX_CHARS = 800
SEMANTIC_CHUNK_SENTENCE_TARGET = 3  # Target sentences per chunk

# Quality thresholds
MIN_TRANSLATION_QUALITY_SCORE = 0.5  # Minimum overall quality score
MAX_RETRY_ATTEMPTS = 2  # Max retries with different backends
SEGMENT_BATCH_DELIMITER = "\n###__SEGMENT_BOUNDARY__###\n"
_TRANSLATOR_SINGLETONS: dict[str, "BaseTranslator"] = {}

FLORES_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
}

# Backend routing configuration
BACKEND_ROUTING = {
    # IndicTrans2 is best for English -> Indic languages
    ("en", "hi"): "indictrans2",
    ("en", "bn"): "indictrans2",
    ("en", "te"): "indictrans2",
    ("en", "ta"): "indictrans2",
    ("en", "mr"): "indictrans2",
    ("en", "gu"): "indictrans2",
    ("en", "kn"): "indictrans2",
    ("en", "ml"): "indictrans2",
    ("en", "pa"): "indictrans2",
    ("en", "ur"): "indictrans2",
    # Indic -> English also good with IndicTrans2
    ("hi", "en"): "indictrans2",
    ("bn", "en"): "indictrans2",
    ("te", "en"): "indictrans2",
    ("ta", "en"): "indictrans2",
    ("mr", "en"): "indictrans2",
    ("gu", "en"): "indictrans2",
    ("kn", "en"): "indictrans2",
    ("ml", "en"): "indictrans2",
    ("pa", "en"): "indictrans2",
    # European languages - MarianMT is good
    ("en", "es"): "marian",
    ("en", "fr"): "marian",
    ("en", "de"): "marian",
    ("es", "en"): "marian",
    ("fr", "en"): "marian",
    ("de", "en"): "marian",
    # Fallback for everything else
}


def _script_profile(text: str) -> dict[str, float]:
    letters = re.findall(r"[A-Za-z\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F]", text)
    total = len(letters) or 1
    latin = len(re.findall(r"[A-Za-z]", text)) / total
    deva = len(re.findall(r"[\u0900-\u097F]", text)) / total
    return {"latin": latin, "deva": deva}


def _should_force_google_auto(src_lang: str, tgt_lang: str, backend: str, text: str) -> bool:
    """
    Romanized Hindi/Hinglish often breaks NLLB/IndicTrans2 when src_lang='hi'
    because those expect Devanagari source. Route to Google(auto) in that case.
    """
    if backend not in {"nllb", "indictrans2"}:
        return False
    if src_lang not in {"hi", "mr"}:
        return False
    if tgt_lang != "en":
        return False
    p = _script_profile(text)
    return p["latin"] >= 0.55 and p["deva"] <= 0.15


def _to_flores(lang: str) -> str:
    code = FLORES_MAP.get(lang)
    if code is None:
        raise ValueError(f"Unknown language code '{lang}' for IndicTrans2.")
    return code


def _chunk_text_simple(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Simple chunking by sentence boundaries."""
    def _split_hard_by_words(raw: str, limit: int) -> list[str]:
        """Guaranteed hard splitter when punctuation boundaries are not useful."""
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if not cleaned:
            return []
        if len(cleaned) <= limit:
            return [cleaned]

        out: list[str] = []
        cur = ""
        for word in cleaned.split(" "):
            if not word:
                continue
            # If a single token is longer than the limit, split that token directly.
            if len(word) > limit:
                if cur:
                    out.append(cur)
                    cur = ""
                for i in range(0, len(word), limit):
                    out.append(word[i : i + limit])
                continue

            candidate = f"{cur} {word}".strip() if cur else word
            if len(candidate) <= limit:
                cur = candidate
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out

    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    # Prefer sentence boundaries first.
    sentence_splits = re.split(r"(?<=[.!?।॥])\s+", normalized)
    sentences = [s.strip() for s in sentence_splits if s and s.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for sent in sentences:
        # Oversized sentence: hard-split and flush.
        if len(sent) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.extend(_split_hard_by_words(sent, max_chars))
            continue

        candidate = f"{current_chunk} {sent}".strip() if current_chunk else sent
        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final safety pass: enforce max length regardless of upstream formatting.
    safe_chunks: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            safe_chunks.append(c)
        else:
            safe_chunks.extend(_split_hard_by_words(c, max_chars))

    return safe_chunks


def _chunk_text_semantic(text: str, segment_boundaries: Optional[list[tuple[float, float]]] = None) -> list[dict]:
    """
    Semantic chunking that preserves meaning boundaries.
    
    Args:
        text: Text to chunk
        segment_boundaries: Optional list of (start, end) timestamps for segments
    
    Returns:
        List of chunk dicts with 'text' and optional 'timing' keys
    """
    if len(text) <= SEMANTIC_CHUNK_MAX_CHARS:
        return [{"text": text, "timing": segment_boundaries[0] if segment_boundaries else None}]
    
    # Split into sentences with their original positions
    sentence_pattern = r'(?<=[.!?।॥])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_timing_start = None
    current_timing_end = None
    sentence_count = 0
    
    for i, sent in enumerate(sentences):
        # Get timing for this sentence if available
        sent_timing = segment_boundaries[i] if segment_boundaries and i < len(segment_boundaries) else None
        
        # Start new chunk if needed
        if not current_chunk:
            current_chunk = sent
            if sent_timing:
                current_timing_start = sent_timing[0]
                current_timing_end = sent_timing[1]
            sentence_count = 1
            continue
        
        # Check if we should add to current chunk
        potential_length = len(current_chunk) + len(sent) + 1
        potential_sentences = sentence_count + 1
        
        # Add to current chunk if:
        # 1. Under max chars AND
        # 2. Under target sentence count OR under min chars
        should_add = (
            potential_length <= SEMANTIC_CHUNK_MAX_CHARS and
            (potential_sentences <= SEMANTIC_CHUNK_SENTENCE_TARGET or
             len(current_chunk) < SEMANTIC_CHUNK_MIN_CHARS)
        )
        
        if should_add:
            current_chunk += " " + sent
            if sent_timing:
                current_timing_end = sent_timing[1]
            sentence_count += 1
        else:
            # Save current chunk and start new one
            chunk_data = {"text": current_chunk.strip()}
            if current_timing_start is not None:
                chunk_data["timing"] = (current_timing_start, current_timing_end)
            chunks.append(chunk_data)
            
            current_chunk = sent
            if sent_timing:
                current_timing_start = sent_timing[0]
                current_timing_end = sent_timing[1]
            sentence_count = 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_data = {"text": current_chunk.strip()}
        if current_timing_start is not None:
            chunk_data["timing"] = (current_timing_start, current_timing_end)
        chunks.append(chunk_data)
    
    return chunks


def _select_backend(src_lang: str, tgt_lang: str, requested_backend: Optional[str] = None) -> str:
    """
    Select the best translation backend for the language pair.
    
    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        requested_backend: User-requested backend (overrides routing if valid)
    
    Returns:
        Backend name to use
    """
    # If user explicitly requested a backend, use it
    if requested_backend:
        valid_backends = ["indictrans2", "marian", "nllb", "google"]
        if requested_backend.lower() in valid_backends:
            log.info("Using user-requested backend: %s", requested_backend)
            return requested_backend.lower()
        log.warning("Invalid backend '%s', using auto-routing", requested_backend)
    
    # Check routing table
    route_key = (src_lang, tgt_lang)
    if route_key in BACKEND_ROUTING:
        backend = BACKEND_ROUTING[route_key]
        log.info("Auto-selected backend for %s->%s: %s", src_lang, tgt_lang, backend)
        return backend
    
    # Check reverse direction
    reverse_key = (tgt_lang, src_lang)
    if reverse_key in BACKEND_ROUTING:
        backend = BACKEND_ROUTING[reverse_key]
        log.info("Auto-selected backend (reverse) for %s->%s: %s", src_lang, tgt_lang, backend)
        return backend
    
    # Default fallback
    default = config.TRANSLATION_BACKEND
    log.info("Using default backend for %s->%s: %s", src_lang, tgt_lang, default)
    return default


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return [self.translate(t, src_lang, tgt_lang) for t in texts]

    def unload(self) -> None:
        self._cleanup_gpu()
    
    def _cleanup_gpu(self):
        """Explicit GPU memory cleanup."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            log.debug("GPU cleanup warning: %s", e)


class IndicTrans2Translator(BaseTranslator):
    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        
        self._requested_device = device
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._oom_fallback = False
        log.info("IndicTrans2Translator device=%s", self._device)

    @lru_cache(maxsize=MAX_GPU_CACHE)
    def _load(self, model_id: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        log.info("Loading IndicTrans2 model: %s", model_id)
        
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self._device)
            model.eval()
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
                
            return tok, model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.error("CUDA OOM loading model. Try: --backend google or restart with empty GPU")
                raise
            raise

    def _pick_model(self, src_lang: str, tgt_lang: str) -> str:
        if src_lang == "en":
            return config.INDICTRANS2_EN_INDIC_MODEL
        if tgt_lang == "en":
            return config.INDICTRANS2_INDIC_EN_MODEL
        raise ValueError(f"IndicTrans2 supports en<->Indic only. Got {src_lang}->{tgt_lang}.")

    def _translate_chunk(self, chunk: str, tok, model, src_f: str, tgt_f: str) -> str:
        """Translate single chunk with OOM recovery."""
        import torch
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                inputs = tok(
                    chunk,
                    src_lang=src_f,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self._device)
                
                with torch.no_grad():
                    ids = model.generate(
                        **inputs,
                        forced_bos_token_id=tok.lang_code_to_id[tgt_f],
                        max_new_tokens=256,
                        num_beams=2,
                        early_stopping=True,
                    )
                
                result = tok.batch_decode(ids, skip_special_tokens=True)[0]
                
                del inputs, ids
                if self._device == "cuda" and not self._oom_fallback:
                    torch.cuda.empty_cache()
                    
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt == 0:
                    log.warning("CUDA OOM during translation, clearing cache and retrying...")
                    self._cleanup_gpu()
                    gc.collect()
                    
                    if self._device == "cuda":
                        log.warning("Switching to CPU for this chunk...")
                        self._oom_fallback = True
                        model = model.cpu()
                        self._device = "cpu"
                        
                        inputs = tok(
                            chunk,
                            src_lang=src_f,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=256,
                        ).to("cpu")
                        
                        with torch.no_grad():
                            ids = model.generate(
                                **inputs,
                                forced_bos_token_id=tok.lang_code_to_id[tgt_f],
                                max_new_tokens=256,
                                num_beams=2,
                                early_stopping=True,
                            )
                        
                        result = tok.batch_decode(ids, skip_special_tokens=True)[0]
                        
                        try:
                            model = model.to(self._requested_device or "cuda")
                            self._device = self._requested_device or "cuda"
                            self._oom_fallback = False
                        except:
                            pass
                            
                        del inputs, ids
                        return result
                raise
        
        return ""

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch
        
        model_id = self._pick_model(src_lang, tgt_lang)
        tok, model = self._load(model_id)
        src_f = _to_flores(src_lang)
        tgt_f = _to_flores(tgt_lang)

        chunks = _chunk_text_simple(text)
        log.info("Translating %d chunks (total %d chars)", len(chunks), len(text))
        
        out_parts: list[str] = []
        
        for i, part in enumerate(chunks):
            log.debug("Translating chunk %d/%d (%d chars)", i+1, len(chunks), len(part))
            translated = self._translate_chunk(part, tok, model, src_f, tgt_f)
            out_parts.append(translated)
            
            if (i + 1) % 3 == 0:  # Cleanup every 3 chunks for 8GB
                self._cleanup_gpu()
        
        self._cleanup_gpu()
        
        return " ".join(out_parts).strip()

    def unload(self) -> None:
        try:
            self._load.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._cleanup_gpu()
        gc.collect()


class MarianMTTranslator(BaseTranslator):
    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_models: dict = {}
        log.info("MarianMTTranslator device=%s", self._device)

    def _load(self, model_id: str):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        log.info("Loading Marian model: %s", model_id)
        
        try:
            tok = MarianTokenizer.from_pretrained(model_id)
            model = MarianMTModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self._device)
            model.eval()
            
            if len(self._loaded_models) >= MAX_GPU_CACHE:
                oldest = next(iter(self._loaded_models))
                del self._loaded_models[oldest]
                self._cleanup_gpu()
            
            self._loaded_models[model_id] = (tok, model)
            return tok, model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.error("CUDA OOM loading Marian model. Falling back to CPU...")
                self._device = "cpu"
                tok = MarianTokenizer.from_pretrained(model_id)
                model = MarianMTModel.from_pretrained(model_id).to("cpu")
                model.eval()
                return tok, model
            raise

    def _model_id(self, src_lang: str, tgt_lang: str) -> str:
        return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    def _translate_with_model(self, text: str, model_id: str) -> str:
        import torch
        
        tok, model = self._load(model_id)
        out_parts: list[str] = []
        
        for part in _chunk_text_simple(text, max_chars=400):
            try:
                inputs = tok([part], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self._device)
                
                with torch.no_grad():
                    ids = model.generate(**inputs, max_length=512)
                
                out_parts.append(tok.decode(ids[0], skip_special_tokens=True))
                del inputs, ids
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning("OOM in Marian, clearing cache...")
                    self._cleanup_gpu()
                    inputs = tok([part], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self._device)
                    with torch.no_grad():
                        ids = model.generate(**inputs, max_length=256)
                    out_parts.append(tok.decode(ids[0], skip_special_tokens=True))
                    del inputs, ids
                else:
                    raise
        
        return " ".join(out_parts).strip()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        direct = self._model_id(src_lang, tgt_lang)
        try:
            result = self._translate_with_model(text, direct)
            log.info("Marian direct used: %s", direct)
            return result
        except OSError:
            if src_lang != "en" and tgt_lang != "en":
                first = self._model_id(src_lang, "en")
                second = self._model_id("en", tgt_lang)
                try:
                    log.warning("No direct Marian model for %s->%s, pivoting via English.", src_lang, tgt_lang)
                    pivot = self._translate_with_model(text, first)
                    result = self._translate_with_model(pivot, second)
                    log.info("Marian pivot used: %s + %s", first, second)
                    return result
                except OSError:
                    raise ValueError(f"No Marian model path for {src_lang}->{tgt_lang}.")
            raise ValueError(f"No Marian model for {src_lang}->{tgt_lang}.")

    def unload(self) -> None:
        self._loaded_models.clear()
        self._cleanup_gpu()
        gc.collect()


class GoogleTranslatorBackend(BaseTranslator):
    _MAX_REQ_CHARS = 4000
    _MAX_REQ_BYTES = 4200
    _MAX_SEGMENTS_PER_GROUP = 24
    _RETRY_ATTEMPTS = 4
    _RETRY_BASE_SLEEP_SEC = 1.0

    def _translate_with_retry(self, tr, text: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self._RETRY_ATTEMPTS + 1):
            try:
                return tr.translate(text)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._RETRY_ATTEMPTS:
                    break
                sleep_s = self._RETRY_BASE_SLEEP_SEC * attempt
                log.warning(
                    "Google translate request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt,
                    self._RETRY_ATTEMPTS,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    def _translate_one(self, tr, text: str) -> str:
        if not text:
            return ""
        if len(text) <= 3000:
            return self._translate_with_retry(tr, text)
        # Deep Translator/Google hard-limits request size; chunk long inputs.
        return " ".join(
            self._translate_with_retry(tr, part)
            for part in _chunk_text_simple(text, max_chars=3000)
        ).strip()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        from deep_translator import GoogleTranslator

        tr = GoogleTranslator(source=src_lang, target=tgt_lang)
        return self._translate_one(tr, text)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self.translate(texts[0], src_lang, tgt_lang)]

        from deep_translator import GoogleTranslator

        tr = GoogleTranslator(source=src_lang, target=tgt_lang)
        outputs = [""] * len(texts)
        marker = "###__SEGMENT_BOUNDARY__###"

        # Build size-safe micro-batches to avoid Google request-size failures.
        groups: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        current_len = 0
        current_bytes = 0

        for idx, text in enumerate(texts):
            t = (text or "").replace(marker, " ").strip()
            if not t:
                outputs[idx] = ""
                continue

            # Very long segment: translate independently with chunking.
            if len(t) >= self._MAX_REQ_CHARS:
                outputs[idx] = self._translate_one(tr, t).strip()
                continue

            # Conservative grouping with both char and byte budgets.
            delim_add_chars = len(SEGMENT_BATCH_DELIMITER) if current else 0
            delim_add_bytes = len(SEGMENT_BATCH_DELIMITER.encode("utf-8")) if current else 0
            add_len = len(t) + delim_add_chars
            add_bytes = len(t.encode("utf-8")) + delim_add_bytes

            would_overflow = (
                current and (
                    current_len + add_len > self._MAX_REQ_CHARS or
                    current_bytes + add_bytes > self._MAX_REQ_BYTES or
                    len(current) >= self._MAX_SEGMENTS_PER_GROUP
                )
            )
            if would_overflow:
                groups.append(current)
                current = [(idx, t)]
                current_len = len(t)
                current_bytes = len(t.encode("utf-8"))
            else:
                current.append((idx, t))
                current_len += add_len
                current_bytes += add_bytes
        if current:
            groups.append(current)

        for group in groups:
            indices = [i for i, _ in group]
            chunks = [t for _, t in group]

            if len(chunks) == 1:
                outputs[indices[0]] = self._translate_one(tr, chunks[0]).strip()
                continue

            # Dynamic boundary token avoids collisions with user text.
            token = f"\n###__SEGMENT_BOUNDARY__{uuid.uuid4().hex}__###\n"
            payload = token.join(chunks)
            try:
                translated = self._translate_with_retry(tr, payload)
                parts = translated.split(token)
            except Exception as exc:
                msg = str(exc)
                if "between 0 and 5000" in msg:
                    log.warning(
                        "Google payload limit hit in group of %d; falling back to per-segment batch API.",
                        len(chunks),
                    )
                else:
                    log.warning(
                        "Google joined-batch call failed for group of %d (%s); falling back.",
                        len(chunks),
                        msg,
                    )
                parts = []

            if len(parts) != len(chunks):
                # Fallback path: let deep_translator do per-item API calls for this group.
                try:
                    alt = tr.translate_batch(chunks)
                    if isinstance(alt, list) and len(alt) == len(chunks):
                        parts = [str(x).strip() for x in alt]
                except Exception:
                    parts = []

            if len(parts) != len(chunks):
                log.warning(
                    "Google batch split mismatch (%d != %d). Falling back to per-segment translation.",
                    len(parts),
                    len(chunks),
                )
                parts = [self._translate_one(tr, t).strip() for t in chunks]

            for idx, out_text in zip(indices, parts):
                outputs[idx] = str(out_text).strip()

        return outputs


class NLLBTranslator(BaseTranslator):
    def __init__(self, device: Optional[str] = None, model_id: str = "facebook/nllb-200-distilled-600M") -> None:
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_id = model_id
        self._loaded = None
        log.info("NLLBTranslator device=%s model=%s", self._device, self._model_id)

    def _load(self):
        if self._loaded is not None:
            return self._loaded

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained(self._model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self._device)
        model.eval()
        self._loaded = (tok, model)
        return self._loaded

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch

        if src_lang not in FLORES_MAP or tgt_lang not in FLORES_MAP:
            raise ValueError(f"NLLB language mapping missing for {src_lang}->{tgt_lang}")

        src_token = _to_flores(src_lang)
        tgt_token = _to_flores(tgt_lang)
        tok, model = self._load()

        out_parts: list[str] = []
        for part in _chunk_text_simple(text, max_chars=1000):
            tok.src_lang = src_token
            inputs = tok(part, return_tensors="pt", truncation=True, max_length=512).to(self._device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    forced_bos_token_id=tok.convert_tokens_to_ids(tgt_token),
                    max_new_tokens=512,
                    num_beams=3,
                )
            out_parts.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
            del inputs, ids
            self._cleanup_gpu()

        return " ".join(out_parts).strip()

    def unload(self) -> None:
        if self._loaded is not None:
            try:
                _, model = self._loaded
                del model
            except Exception:
                pass
            self._loaded = None
        self._cleanup_gpu()
        gc.collect()


def _resolve_backend_and_src(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: Optional[str],
) -> tuple[str, str]:
    selected_backend = _select_backend(src_lang, tgt_lang, backend)
    resolved_src = src_lang
    # Detect Romanized source BEFORE any heavy model instantiation.
    if _should_force_google_auto(src_lang, tgt_lang, selected_backend, text):
        log.warning(
            "Detected Romanized %s input (latin-heavy). "
            "Overriding backend %s -> google(auto) for better translation.",
            src_lang,
            selected_backend,
        )
        selected_backend = "google"
        resolved_src = "auto"
    return selected_backend, resolved_src


def get_translator(backend: Optional[str] = None) -> BaseTranslator:
    name = (backend or config.TRANSLATION_BACKEND).lower()
    if name in _TRANSLATOR_SINGLETONS:
        return _TRANSLATOR_SINGLETONS[name]
    if name == "indictrans2":
        tr: BaseTranslator = IndicTrans2Translator()
    elif name == "marian":
        tr = MarianMTTranslator()
    elif name == "nllb":
        tr = NLLBTranslator()
    elif name == "google":
        tr = GoogleTranslatorBackend()
    else:
        raise ValueError(f"Unknown translation backend: {name}")
    _TRANSLATOR_SINGLETONS[name] = tr
    return tr


def unload_translation_model() -> None:
    """
    Release translation backend resources (especially NLLB GPU memory)
    between pipeline stages.
    """
    for _, translator in list(_TRANSLATOR_SINGLETONS.items()):
        try:
            translator.unload()
        except Exception as exc:
            log.debug("Translator unload warning: %s", exc)
    _TRANSLATOR_SINGLETONS.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def translate_with_quality_gate(
    text: str,
    tgt_lang: str,
    src_lang: str = "en",
    backend: Optional[str] = None,
    strict: bool = True,
    use_glossary: bool = True,
) -> tuple[str, ValidationResult]:
    """
    Translate with pre/post quality gates and glossary support.
    
    Args:
        text: Source text to translate
        tgt_lang: Target language code
        src_lang: Source language code
        backend: Translation backend to use
        strict: If True, fail on low quality output
        use_glossary: If True, apply glossary term locking
    
    Returns:
        Tuple of (translated_text, validation_result)
    """
    if not text.strip():
        raise ValueError("translate() received empty text.")
    
    # Pre-translation: Lock glossary terms
    placeholder_map = {}
    if use_glossary:
        text_with_placeholders, placeholder_map = lock_terms(text)
        if placeholder_map:
            log.info("Locked %d glossary terms", len(placeholder_map))
    else:
        text_with_placeholders = text
    
    # Select backend before model load
    selected_backend, src_lang = _resolve_backend_and_src(
        text_with_placeholders,
        src_lang,
        tgt_lang,
        backend,
    )
    translator = get_translator(selected_backend)
    
    log.info("Translating %d chars [%s->%s] using %s", 
             len(text_with_placeholders), src_lang, tgt_lang, 
             translator.__class__.__name__)
    
    # Perform translation
    try:
        result = translator.translate(text_with_placeholders, src_lang, tgt_lang)
    except Exception as e:
        log.error("Translation failed with %s: %s", selected_backend, e)
        # Try fallback
        if selected_backend != "google":
            log.warning("Falling back to Google Translate...")
            return translate_with_quality_gate(text, tgt_lang, src_lang, "google", strict, use_glossary)
        raise
    
    # Post-translation: Restore glossary terms
    if use_glossary and placeholder_map:
        result = restore_terms(result, placeholder_map, tgt_lang)
    
    # Validate output
    validation = validate_translation(text, result, src_lang, tgt_lang, strict)
    
    if not validation.passed and strict:
        log.error("Translation quality check failed: %s", validation.reason)

        # Common failure mode for Hinglish / mixed-script source:
        # forced src_lang can cause near-copy outputs. Retry once with auto detect.
        if (
            "Near-source copy detected" in validation.reason
            and selected_backend == "google"
            and src_lang != "auto"
        ):
            log.warning("Retrying Google translation with src_lang=auto due to near-copy detection...")
            return translate_with_quality_gate(
                text,
                tgt_lang,
                "auto",
                "google",
                strict,
                use_glossary,
            )

        # Try retry with different backend
        if selected_backend != "google":
            log.warning("Retrying with Google Translate...")
            return translate_with_quality_gate(text, tgt_lang, src_lang, "google", strict, use_glossary)
        
        raise RuntimeError(f"Translation quality too low: {validation.reason}")
    
    if not validation.passed:
        log.warning("Translation quality warning: %s", validation.reason)
    
    # Verify glossary terms
    if use_glossary:
        issues = verify_terms(text, result, tgt_lang)
        if issues:
            log.warning("Glossary verification found %d issues", len(issues))
            for issue in issues[:3]:
                log.warning("  %s: '%s' -> expected '%s'", 
                          issue["type"], issue["original"], issue["expected"])
    
    return result, validation


def translate(
    text: str,
    tgt_lang: str,
    src_lang: str = "en",
    backend: Optional[str] = None,
    use_glossary: bool = True,
) -> str:
    """
    Simple translate function (backward compatible).
    
    Args:
        text: Source text
        tgt_lang: Target language code
        src_lang: Source language code
        backend: Translation backend
        use_glossary: Apply glossary locking
    
    Returns:
        Translated text
    """
    result, _ = translate_with_quality_gate(text, tgt_lang, src_lang, backend, strict=False, use_glossary=use_glossary)
    return result


def translate_texts_batch(
    texts: list[str],
    tgt_lang: str,
    src_lang: str = "en",
    backend: Optional[str] = None,
    use_glossary: bool = True,
) -> list[str]:
    """
    Batch translate multiple short texts efficiently.
    Uses one backend call when possible (Google backend supports delimiter batching).
    """
    if not texts:
        return []

    outputs = [""] * len(texts)
    indexed_non_empty = [(i, t) for i, t in enumerate(texts) if (t or "").strip()]
    if not indexed_non_empty:
        return outputs

    ordered_texts = [t.strip() for _, t in indexed_non_empty]

    placeholder_maps: list[dict[str, str]] = []
    locked_texts: list[str] = []
    if use_glossary:
        for t in ordered_texts:
            locked, pmap = lock_terms(t)
            locked_texts.append(locked)
            placeholder_maps.append(pmap)
    else:
        locked_texts = ordered_texts
        placeholder_maps = [{} for _ in ordered_texts]

    joined_for_detection = " ".join(locked_texts)
    selected_backend, resolved_src = _resolve_backend_and_src(
        joined_for_detection,
        src_lang,
        tgt_lang,
        backend,
    )
    translator = get_translator(selected_backend)

    try:
        translated_locked = translator.translate_batch(locked_texts, resolved_src, tgt_lang)
    except Exception as exc:
        log.error("Batch translation failed with %s: %s", selected_backend, exc)
        if selected_backend != "google":
            log.warning("Falling back to Google batch translation...")
            return translate_texts_batch(
                texts,
                tgt_lang=tgt_lang,
                src_lang=src_lang,
                backend="google",
                use_glossary=use_glossary,
            )
        raise

    if len(translated_locked) != len(locked_texts):
        log.warning(
            "Batch output size mismatch (%d != %d). Falling back to per-text translation.",
            len(translated_locked),
            len(locked_texts),
        )
        translated_locked = [translator.translate(t, resolved_src, tgt_lang) for t in locked_texts]

    restored: list[str] = []
    for out, pmap, source_text in zip(translated_locked, placeholder_maps, ordered_texts):
        restored_text = restore_terms(out, pmap, tgt_lang) if use_glossary and pmap else out
        validation = validate_translation(source_text, restored_text, resolved_src, tgt_lang, strict=False)
        if not validation.passed:
            log.debug("Batch translation quality warning: %s", validation.reason)
        restored.append(restored_text.strip())

    for (orig_idx, _), translated_text in zip(indexed_non_empty, restored):
        outputs[orig_idx] = translated_text
    return outputs


def translate_segments_with_semantic_chunking(
    segments: list[dict],
    tgt_lang: str,
    src_lang: str = "en",
    backend: Optional[str] = None,
    use_glossary: bool = True,
) -> list[dict]:
    """
    Translate segments with semantic chunking for better context preservation.
    
    Args:
        segments: List of segments with 'text', 'start', 'end' keys
        tgt_lang: Target language code
        src_lang: Source language code
        backend: Translation backend
        use_glossary: Apply glossary locking
    
    Returns:
        List of translated segments with same structure
    """
    if not segments:
        return []
    
    # Combine segments for semantic chunking
    full_text = " ".join(s.get("text", "") for s in segments)
    boundaries = [(s.get("start", 0), s.get("end", 0)) for s in segments]
    
    # Create semantic chunks
    chunks = _chunk_text_semantic(full_text, boundaries)
    
    log.info("Semantic chunking: %d segments -> %d chunks", len(segments), len(chunks))
    
    # Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        try:
            translated, validation = translate_with_quality_gate(
                chunk["text"],
                tgt_lang,
                src_lang,
                backend,
                strict=True,
                use_glossary=use_glossary,
            )
        except Exception as exc:
            log.warning("Strict chunk translation failed (%s). Retrying non-strict with smaller chunks...", exc)
            small_parts = _chunk_text_simple(chunk["text"], max_chars=220)
            translated_parts: list[str] = []
            scores: list[float] = []
            for part in small_parts:
                part_translated, part_validation = translate_with_quality_gate(
                    part,
                    tgt_lang,
                    src_lang,
                    backend,
                    strict=False,
                    use_glossary=use_glossary,
                )
                translated_parts.append(part_translated)
                scores.append(part_validation.score)

            translated = " ".join(p for p in translated_parts if p).strip()
            validation = ValidationResult(
                passed=bool(translated),
                score=(sum(scores) / len(scores)) if scores else 0.0,
                reason="Recovered via non-strict sub-chunk translation",
                details={"sub_chunks": len(small_parts)},
            )

        if not validation.passed:
            log.warning("Chunk translation quality warning: %s", validation.reason)

        translated_chunks.append({
            "text": translated,
            "timing": chunk.get("timing"),
            "quality_score": validation.score,
        })
    
    # Map back to segment structure
    # For now, return chunks as segments (may need refinement)
    result = []
    for chunk in translated_chunks:
        timing = chunk.get("timing")
        result.append({
            "start": timing[0] if timing else 0,
            "end": timing[1] if timing else 0,
            "text": chunk["text"],
            "quality_score": chunk.get("quality_score", 0),
        })
    
    return result
