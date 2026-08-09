"""
glossary.py
-----------
Domain glossary and terminology management for consistent translation.
Ensures brand names, technical terms, and auto terminology are locked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config

log = config.get_logger(__name__)


@dataclass
class TermEntry:
    """A glossary term entry with translations and metadata."""
    term: str  # Original term (English/source)
    translations: dict[str, str] = field(default_factory=dict)  # lang_code -> translation
    category: str = "general"  # brand, technical, automotive, etc.
    transliteration: dict[str, str] = field(default_factory=dict)  # For script-based languages
    locked: bool = True  # If True, always use specified translation
    case_sensitive: bool = False
    whole_word: bool = True  # Match whole word only


class DomainGlossary:
    """
    Manages domain-specific terminology for consistent translation.
    
    Features:
    - Pre-translation term extraction and locking
    - Post-translation term verification
    - Support for both translation and transliteration
    - Category-based term organization
    """
    
    def __init__(self, glossary_path: Optional[str | Path] = None):
        self.terms: dict[str, TermEntry] = {}
        self.categories: set[str] = set()
        self._glossary_path = glossary_path
        
        # Load built-in automotive glossary
        self._load_builtin_glossary()
        
        # Load custom glossary if provided
        if glossary_path:
            self.load_from_file(glossary_path)
    
    def _load_builtin_glossary(self) -> None:
        """Load built-in automotive glossary for Hindi content."""
        builtin_terms = [
            # Brand names - always locked, never translate
            TermEntry("Maruti Suzuki", {"hi": "Maruti Suzuki"}, "brand", locked=True),
            TermEntry("Tata Motors", {"hi": "Tata Motors"}, "brand", locked=True),
            TermEntry("Mahindra", {"hi": "Mahindra"}, "brand", locked=True),
            TermEntry("Hyundai", {"hi": "Hyundai"}, "brand", locked=True),
            TermEntry("Toyota", {"hi": "Toyota"}, "brand", locked=True),
            TermEntry("Honda", {"hi": "Honda"}, "brand", locked=True),
            TermEntry("Kia", {"hi": "Kia"}, "brand", locked=True),
            TermEntry("MG", {"hi": "MG"}, "brand", locked=True),
            TermEntry("Skoda", {"hi": "Skoda"}, "brand", locked=True),
            TermEntry("Volkswagen", {"hi": "Volkswagen"}, "brand", locked=True),
            TermEntry("BMW", {"hi": "BMW"}, "brand", locked=True),
            TermEntry("Mercedes", {"hi": "Mercedes"}, "brand", locked=True),
            TermEntry("Audi", {"hi": "Audi"}, "brand", locked=True),
            TermEntry("Jeep", {"hi": "Jeep"}, "brand", locked=True),
            TermEntry("Ford", {"hi": "Ford"}, "brand", locked=True),
            TermEntry("Nissan", {"hi": "Nissan"}, "brand", locked=True),
            TermEntry("Renault", {"hi": "Renault"}, "brand", locked=True),
            
            # Model names - locked
            TermEntry("Swift", {"hi": "Swift"}, "brand", locked=True),
            TermEntry("Baleno", {"hi": "Baleno"}, "brand", locked=True),
            TermEntry("WagonR", {"hi": "WagonR"}, "brand", locked=True),
            TermEntry("Alto", {"hi": "Alto"}, "brand", locked=True),
            TermEntry("Brezza", {"hi": "Brezza"}, "brand", locked=True),
            TermEntry("Creta", {"hi": "Creta"}, "brand", locked=True),
            TermEntry("Venue", {"hi": "Venue"}, "brand", locked=True),
            TermEntry("Nexon", {"hi": "Nexon"}, "brand", locked=True),
            TermEntry("Punch", {"hi": "Punch"}, "brand", locked=True),
            TermEntry("Harrier", {"hi": "Harrier"}, "brand", locked=True),
            TermEntry("Safari", {"hi": "Safari"}, "brand", locked=True),
            TermEntry("Scorpio", {"hi": "Scorpio"}, "brand", locked=True),
            TermEntry("Thar", {"hi": "Thar"}, "brand", locked=True),
            TermEntry("XUV700", {"hi": "XUV700"}, "brand", locked=True),
            TermEntry("Bolero", {"hi": "Bolero"}, "brand", locked=True),
            
            # Technical terms - translate consistently
            TermEntry("engine", {"hi": "इंजन"}, "technical", locked=True),
            TermEntry("transmission", {"hi": "ट्रांसमिशन"}, "technical", locked=True),
            TermEntry("mileage", {"hi": "माइलेज"}, "technical", locked=True),
            TermEntry("fuel efficiency", {"hi": "ईंधन दक्षता"}, "technical", locked=True),
            TermEntry("horsepower", {"hi": "हॉर्सपावर"}, "technical", locked=True),
            TermEntry("torque", {"hi": "टॉर्क"}, "technical", locked=True),
            TermEntry("suspension", {"hi": "सस्पेंशन"}, "technical", locked=True),
            TermEntry("brakes", {"hi": "ब्रेक"}, "technical", locked=True),
            TermEntry("airbags", {"hi": "एयरबैग"}, "technical", locked=True),
            TermEntry("ABS", {"hi": "ABS"}, "technical", locked=True),
            TermEntry("ESP", {"hi": "ESP"}, "technical", locked=True),
            TermEntry("infotainment", {"hi": "इंफोटेनमेंट"}, "technical", locked=True),
            TermEntry("touchscreen", {"hi": "टचस्क्रीन"}, "technical", locked=True),
            TermEntry("sunroof", {"hi": "सनरूफ"}, "technical", locked=True),
            TermEntry("boot space", {"hi": "बूट स्पेस"}, "technical", locked=True),
            TermEntry("ground clearance", {"hi": "ग्राउंड क्लीयरेंस"}, "technical", locked=True),
            TermEntry("wheelbase", {"hi": "व्हीलबेस"}, "technical", locked=True),
            TermEntry("turbo", {"hi": "टर्बो"}, "technical", locked=True),
            TermEntry("automatic", {"hi": "ऑटोमैटिक"}, "technical", locked=True),
            TermEntry("manual", {"hi": "मैनुअल"}, "technical", locked=True),
            TermEntry("CVT", {"hi": "CVT"}, "technical", locked=True),
            TermEntry("DCT", {"hi": "DCT"}, "technical", locked=True),
            TermEntry("AMT", {"hi": "AMT"}, "technical", locked=True),
            
            # Units - transliterate
            TermEntry("kmpl", {"hi": "किलोमीटर प्रति लीटर"}, "unit", locked=True),
            TermEntry("km/l", {"hi": "किलोमीटर प्रति लीटर"}, "unit", locked=True),
            TermEntry("kmph", {"hi": "किलोमीटर प्रति घंटा"}, "unit", locked=True),
            TermEntry("km/h", {"hi": "किलोमीटर प्रति घंटा"}, "unit", locked=True),
            TermEntry("cc", {"hi": "सीसी"}, "unit", locked=True),
            TermEntry("litre", {"hi": "लीटर"}, "unit", locked=True),
            TermEntry("bhp", {"hi": "बीएचपी"}, "unit", locked=True),
            TermEntry("Nm", {"hi": "एनएम"}, "unit", locked=True),
            TermEntry("mm", {"hi": "एमएम"}, "unit", locked=True),
            
            # Price terms
            TermEntry("lakh", {"hi": "लाख"}, "price", locked=True),
            TermEntry("crore", {"hi": "करोड़"}, "price", locked=True),
            TermEntry("on-road price", {"hi": "ऑन-रोड कीमत"}, "price", locked=True),
            TermEntry("ex-showroom", {"hi": "एक्स-शोरूम"}, "price", locked=True),
            
            # Common phrases - conversational Hindi
            TermEntry("test drive", {"hi": "टेस्ट ड्राइव"}, "general", locked=True),
            TermEntry("service center", {"hi": "सर्विस सेंटर"}, "general", locked=True),
            TermEntry("warranty", {"hi": "वारंटी"}, "general", locked=True),
            TermEntry("insurance", {"hi": "बीमा"}, "general", locked=True),
            TermEntry("EMI", {"hi": "EMI"}, "general", locked=True),
            TermEntry("down payment", {"hi": "डाउन पेमेंट"}, "general", locked=True),
            TermEntry("exchange", {"hi": "एक्सचेंज"}, "general", locked=True),
            TermEntry("resale value", {"hi": "रिसेल वैल्यू"}, "general", locked=True),
        ]
        
        for entry in builtin_terms:
            self.add_term(entry)
        
        log.info("Loaded %d built-in glossary terms", len(builtin_terms))
    
    def add_term(self, entry: TermEntry) -> None:
        """Add a term to the glossary."""
        key = entry.term.lower() if not entry.case_sensitive else entry.term
        self.terms[key] = entry
        self.categories.add(entry.category)
    
    def get_term(self, term: str, case_sensitive: bool = False) -> Optional[TermEntry]:
        """Get a term entry by its original form."""
        key = term if case_sensitive else term.lower()
        return self.terms.get(key)
    
    def get_translation(self, term: str, lang: str, case_sensitive: bool = False) -> Optional[str]:
        """Get the translation for a term in a specific language."""
        entry = self.get_term(term, case_sensitive)
        if entry and entry.locked:
            return entry.translations.get(lang)
        return None
    
    def load_from_file(self, path: str | Path) -> None:
        """Load glossary from JSON file."""
        path = Path(path)
        if not path.exists():
            log.warning("Glossary file not found: %s", path)
            return
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("terms", []):
                entry = TermEntry(
                    term=item["term"],
                    translations=item.get("translations", {}),
                    category=item.get("category", "general"),
                    transliteration=item.get("transliteration", {}),
                    locked=item.get("locked", True),
                    case_sensitive=item.get("case_sensitive", False),
                    whole_word=item.get("whole_word", True),
                )
                self.add_term(entry)
            
            log.info("Loaded %d terms from %s", len(data.get("terms", [])), path)
        except Exception as e:
            log.error("Failed to load glossary from %s: %s", path, e)
    
    def save_to_file(self, path: str | Path) -> None:
        """Save glossary to JSON file."""
        path = Path(path)
        data = {
            "terms": [
                {
                    "term": entry.term,
                    "translations": entry.translations,
                    "category": entry.category,
                    "transliteration": entry.transliteration,
                    "locked": entry.locked,
                    "case_sensitive": entry.case_sensitive,
                    "whole_word": entry.whole_word,
                }
                for entry in self.terms.values()
            ]
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Saved %d terms to %s", len(self.terms), path)
    
    def extract_terms_from_text(self, text: str) -> list[tuple[str, TermEntry]]:
        """
        Extract glossary terms found in text.
        
        Returns:
            List of (matched_text, term_entry) tuples
        """
        found = []
        text_lower = text.lower()
        
        for key, entry in self.terms.items():
            pattern = self._make_pattern(entry)
            for match in re.finditer(pattern, text, re.IGNORECASE if not entry.case_sensitive else 0):
                found.append((match.group(), entry))
        
        # Sort by position in text for consistent replacement
        found.sort(key=lambda x: text_lower.find(x[0].lower()))
        return found
    
    def _make_pattern(self, entry: TermEntry) -> str:
        """Create regex pattern for matching a term."""
        term = re.escape(entry.term)
        if entry.whole_word:
            return r"\b" + term + r"\b"
        return term
    
    def lock_terms_before_translation(
        self,
        text: str,
        placeholder_format: str = "<<{category}:{index}>>"
    ) -> tuple[str, dict[str, str]]:
        """
        Replace glossary terms with placeholders before translation.
        
        Args:
            text: Original text
            placeholder_format: Format for placeholders
        
        Returns:
            Tuple of (text_with_placeholders, placeholder_map)
        """
        found_terms = self.extract_terms_from_text(text)
        placeholder_map = {}
        category_counters = {}
        
        result = text
        
        for matched_text, entry in found_terms:
            if not entry.locked:
                continue
            
            # Generate placeholder
            cat = entry.category
            idx = category_counters.get(cat, 0)
            placeholder = placeholder_format.format(category=cat, index=idx)
            category_counters[cat] = idx + 1
            
            # Store mapping
            placeholder_map[placeholder] = matched_text
            
            # Replace in text
            pattern = self._make_pattern(entry)
            result = re.sub(pattern, placeholder, result, count=1, 
                          flags=re.IGNORECASE if not entry.case_sensitive else 0)
        
        return result, placeholder_map
    
    def restore_terms_after_translation(
        self,
        text: str,
        placeholder_map: dict[str, str],
        lang: str
    ) -> str:
        """
        Restore glossary terms after translation.
        
        Args:
            text: Translated text with placeholders
            placeholder_map: Mapping of placeholders to original terms
            lang: Target language code
        
        Returns:
            Text with glossary terms properly translated
        """
        result = text
        
        for placeholder, original_term in placeholder_map.items():
            entry = self.get_term(original_term)
            
            if entry and lang in entry.translations:
                # Use locked translation
                replacement = entry.translations[lang]
            else:
                # Keep original if no translation available
                replacement = original_term
            
            result = result.replace(placeholder, replacement)
        
        return result
    
    def verify_translation(
        self,
        source: str,
        translated: str,
        lang: str
    ) -> list[dict]:
        """
        Verify that glossary terms are correctly translated.
        
        Returns:
            List of issues found
        """
        issues = []
        found_terms = self.extract_terms_from_text(source)
        
        for matched_text, entry in found_terms:
            if not entry.locked:
                continue
            
            expected = entry.translations.get(lang)
            if not expected:
                continue
            
            # Check if expected translation is in output
            if expected.lower() not in translated.lower():
                # Check if original term is still there (bad)
                if matched_text.lower() in translated.lower():
                    issues.append({
                        "type": "untranslated_term",
                        "original": matched_text,
                        "expected": expected,
                        "severity": "error"
                    })
                else:
                    issues.append({
                        "type": "missing_term",
                        "original": matched_text,
                        "expected": expected,
                        "severity": "warning"
                    })
        
        return issues


# Global glossary instance
_glossary: Optional[DomainGlossary] = None


def get_glossary(glossary_path: Optional[str | Path] = None) -> DomainGlossary:
    """Get singleton glossary instance."""
    global _glossary
    if _glossary is None:
        _glossary = DomainGlossary(glossary_path)
    return _glossary


def lock_terms(text: str) -> tuple[str, dict[str, str]]:
    """Convenience function to lock terms before translation."""
    return get_glossary().lock_terms_before_translation(text)


def restore_terms(text: str, placeholder_map: dict[str, str], lang: str) -> str:
    """Convenience function to restore terms after translation."""
    return get_glossary().restore_terms_after_translation(text, placeholder_map, lang)


def verify_terms(source: str, translated: str, lang: str) -> list[dict]:
    """Convenience function to verify term translation."""
    return get_glossary().verify_translation(source, translated, lang)
