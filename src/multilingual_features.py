"""Unicode-aware lexical features.

Han characters and Latin tokens are counted as heuristic evidence units, not
linguistic words. The language tag is descriptive metadata, not a business feature.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES = [
    "char_cosine", "char_ngram_jaccard", "length_ratio",
    "length_difference_normalized", "exact_match", "short_evidence", "generic_phrase",
    "negation_conflict", "numeric_unit_conflict", "model_conflict",
]
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[a-z]+(?:['’][a-z]+)?", re.I)
_NEG = re.compile(r"\b(?:not|no|never|cannot|can't|doesn't|don't|didn't|without|isn't|aren't|wasn't|weren't|won't|unsupported|incompatible)\b|(?:不支持|不兼容|不能|无法|没有|并非|不是|未能|不可以|不带|无)", re.I)
_POS = re.compile(r"\b(?:supports?|compatible|works?)(?:\s+with)?\b|\b(?:yes|can)\b|支持|兼容|可以|能", re.I)
_ATTRS = {
    "PD support": r"(?<![A-Za-z0-9_])PD(?![A-Za-z0-9_])|\bpower delivery\b|PD快充|PD协议",
    "battery life": r"battery life|battery lasts?|续航|电池.{0,5}(?:耐用|持久|续航)",
    "screen cracking": r"screen.{0,12}(?:crack|break)|(?:crack|break).{0,12}screen|屏幕.{0,5}(?:裂|碎)|(?:裂|碎).{0,5}屏幕",
    "charging speed": r"fast charg|charge(?:s|d|ing)? fast|slow charg|charge.{0,20}slow|slow.{0,20}charg|充电.{0,4}(?:快|慢)|快充|慢充",
    "water resistance": r"water.?proof|water resistant|防水|防水性能",
    "data transfer": r"data transfer|transfer files?|传输文件|传数据|数据传输",
    "magnetic attachment": r"magsafe|magnetic (?:ring|attachment)|磁吸|磁力环",
}
_NUM_CUES = {
    "charging power": r"\b(?:charge|charging|charger|power)\b|充电|快充|功率",
    "battery capacity": r"\bbattery\b|\bpower bank\b|电池|充电宝|容量",
    "battery duration": r"battery life|battery lasts?|续航|待机|使用时间",
    "screen size": r"screen|display|inch(?:es)?|屏幕|显示屏",
}
# Explicit units only. ASCII boundaries allow adjacent Han characters: 65W瓦.
_NUM = re.compile(r"(?<![A-Za-z0-9_])([+-]?\d+(?:[.,]\d+)?)\s*(mah|kw|wh|w|h|hours?|hrs?|minutes?|mins?|days?|%|hz|gb|tb|mm|cm|inches|inch|in)(?![A-Za-z0-9_])", re.I)
_SENTENCE = re.compile(r"[.!?。！？;；,，\n]+")
_GENERIC_ZH = re.compile(r"(?:非常好|很好用|很好|不错|值得购买|物超所值|质量很好|满意|推荐|好评)")
_DETAIL_ZH = re.compile(r"(?:因为|由于|但是|不过|例如|比如|充电|电池|屏幕|安装|尺寸|声音|按键|材质|做工|续航|速度|重量|接口|功能|缺点|问题|使用了|用了\s*\d|\d+\s*(?:天|小时|分钟|次|元|块))")

def normalize_text(text: str) -> str:
    """NFKC/casefold/whitespace normalization; preserve negatives and IDs."""
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())

def analyze_text(text: str) -> dict:
    n = normalize_text(text)
    han = len(_HAN.findall(n))
    latin = _LATIN.findall(n)
    units = han + len(latin)
    generic = bool(_GENERIC_ZH.search(n)) and not bool(_DETAIL_ZH.search(n))
    return {
        "normalized": n,
        "language": "mixed" if han and latin else "zh" if han else "en" if latin else "other",
        "han_characters": han,
        "latin_tokens": len(latin),
        "effective_length": units,
        "short_evidence": units < 5,
        "generic_phrase": generic,
    }

def _segments(text):
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]

def _polarities(text):
    result = {}
    for attr, cue in _ATTRS.items():
        states = set()
        for segment in _segments(text):
            if not re.search(cue, segment, re.I):
                continue
            neg = bool(_NEG.search(segment))
            # Treat explicit slow/negative speed wording as negative evidence.
            slow = bool(re.search(r"\bslow(?:ly)?\b|慢", segment, re.I)) if attr == "charging speed" else False
            fast = bool(re.search(r"\bfast\b|快", segment, re.I)) if attr == "charging speed" else False
            if neg or slow:
                states.add(-1)
            elif fast or _POS.search(segment):
                states.add(1)
        if len(states) == 1:
            result[attr] = next(iter(states))
    return result

def _unit_value(attr, value, unit):
    """Convert compatible explicit units to canonical values; None means incomparable."""
    unit = unit.casefold()
    if attr == "charging power" and unit in {"w", "kw"}:
        return value * (1000 if unit == "kw" else 1), "w"
    if attr == "battery capacity" and unit in {"mah", "wh"}:
        return value, unit  # mAh and Wh require voltage for conversion.
    if attr == "battery duration":
        if unit in {"h", "hr", "hrs", "hour", "hours"}: return value * 60, "min"
        if unit in {"min", "mins", "minute", "minutes"}: return value, "min"
        if unit == "days": return value * 1440, "min"
    if attr == "screen size":
        if unit in {"in", "inch", "inches"}: return value * 25.4, "mm"
        if unit == "cm": return value * 10, "mm"
        if unit == "mm": return value, "mm"
    return None

def _numeric_claims(text):
    claims = {}
    for segment in _segments(text):
        cues = [(m.start(), m.end(), attr) for attr, pattern in _NUM_CUES.items()
                for m in re.finditer(pattern, segment, re.I)]
        for m in _NUM.finditer(segment):
            try:
                value = float(m.group(1).replace(",", "."))
            except ValueError:
                continue
            unit = m.group(2).casefold()
            nearest = [(min(abs(m.start()-start), abs(m.start()-end)), attr)
                       for start, end, attr in cues]
            if not nearest:
                continue
            distance, attr = min(nearest)
            if distance > 48:
                continue
            canonical = _unit_value(attr, value, unit)
            if canonical is not None:
                claims.setdefault(attr, canonical)
    return claims

def _models(text):
    found = {}
    patterns = [
        r"\b(iphone|ipad|galaxy|pixel|thinkpad|macbook)\s*[- ]?([a-z]?\d+[a-z]?)\b",
        r"\b(model|型号)\s*[:#-]?\s*([a-z0-9][a-z0-9._-]{1,})",
    ]
    for i, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text, re.I):
            key, value = ("model", match.group(2)) if i else (match.group(1).casefold(), match.group(2))
            found[key] = value.casefold()
    return found

def conflict_evidence(text_a: str, text_b: str) -> dict:
    """Return contextual candidate cues for review, not a contradiction verdict."""
    a, b = normalize_text(text_a), normalize_text(text_b)
    pa, pb = _polarities(a), _polarities(b)
    neg = sorted(k for k in pa.keys() & pb.keys() if pa[k] != pb[k])
    na, nb = _numeric_claims(a), _numeric_claims(b)
    numeric = []
    for key in sorted(na.keys() & nb.keys()):
        if na[key][1] == nb[key][1] and not np.isclose(na[key][0], nb[key][0]):
            numeric.append({"attribute": key, "claim_a": f"{na[key][0]:g}{na[key][1]}",
                            "claim_b": f"{nb[key][0]:g}{nb[key][1]}"})
    ma, mb = _models(a), _models(b)
    models = [{"family": k, "model_a": ma[k], "model_b": mb[k]}
              for k in sorted(ma.keys() & mb.keys()) if ma[k] != mb[k]]
    return {
        "negation_conflicts": neg, "numeric_unit_conflicts": numeric,
        "model_conflicts": models, "candidate_conflict": bool(neg or numeric or models),
        "note": "Conservative candidate cues for review; not a guaranteed contradiction.",
    }

class MultilingualLexical:
    def __init__(self):
        self.vectorizer = None

    def fit(self, texts: Iterable[str]):
        corpus = [normalize_text(t) for t in texts]
        if not corpus or not any(corpus):
            raise ValueError("Cannot fit char TF-IDF: provide non-empty training texts.")
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=2,
            max_features=100000, dtype=np.float32,
        )
        try:
            self.vectorizer.fit(corpus)
        except ValueError as exc:
            if "empty vocabulary" in str(exc) or "no terms remain" in str(exc):
                raise ValueError("Cannot fit char TF-IDF: no character n-grams occur at least twice in training texts.") from exc
            raise
        return self

    def transform_pairs(self, text_a: Sequence[str], text_b: Sequence[str], semantic_scores=None):
        if self.vectorizer is None:
            raise RuntimeError("Call fit(texts) first.")
        if len(text_a) != len(text_b):
            raise ValueError("Paired text lists must have equal lengths.")
        if not text_a:
            return np.empty((0, len(FEATURE_NAMES) + (semantic_scores is not None)), dtype=np.float32)
        aa, bb = [normalize_text(x) for x in text_a], [normalize_text(x) for x in text_b]
        xa, xb = self.vectorizer.transform(aa), self.vectorizer.transform(bb)
        cosine = np.asarray(xa.multiply(xb).sum(axis=1)).ravel()
        rows = []
        for i, (a, b) in enumerate(zip(aa, bb)):
            ta, tb = analyze_text(a), analyze_text(b)
            ga = {a[j:j+n] for n in range(2, 5) for j in range(max(0, len(a)-n+1))}
            gb = {b[j:j+n] for n in range(2, 5) for j in range(max(0, len(b)-n+1))}
            union = ga | gb
            jaccard = len(ga & gb) / len(union) if union else float(a == b)
            la, lb = ta["effective_length"], tb["effective_length"]
            ev = conflict_evidence(a, b)
            rows.append([
                float(cosine[i]), jaccard, min(la, lb)/max(la, lb) if max(la, lb) else 1.,
                abs(la-lb)/max(la, lb, 1), float(a == b),
                float(ta["short_evidence"] or tb["short_evidence"]),float(ta["generic_phrase"] or tb["generic_phrase"]),
                float(bool(ev["negation_conflicts"])), float(bool(ev["numeric_unit_conflicts"])),
                float(bool(ev["model_conflicts"])),
            ])
        matrix = np.asarray(rows, dtype=np.float32)
        if semantic_scores is not None:
            scores = np.asarray(semantic_scores, dtype=np.float32).reshape(-1, 1)
            if len(scores) != len(matrix):
                raise ValueError("semantic_scores length must match paired texts.")
            matrix = np.column_stack([matrix, scores])
        return matrix


def pair_language(a,b):
    tags={analyze_text(t)["language"] for t in (a,b)}
    if "mixed" in tags or ("zh" in tags and "en" in tags): return "mixed"
    if "zh" in tags: return "zh"
    if "en" in tags: return "en"
    return "other"


def score_methods(x, models):
    return {"A":x[:,FEATURE_NAMES.index("exact_match")],"B":x[:,0],
            "C":models["C"].predict_proba(x[:,:len(FEATURE_NAMES)])[:,1],
            "D":x[:,-1],"E":models["E"].predict_proba(x)[:,1]}


def allowed_features(x):
    names={name:i for i,name in enumerate(FEATURE_NAMES)}
    flags=["short_evidence","generic_phrase","negation_conflict","numeric_unit_conflict","model_conflict"]
    return ~x[:,[names[f] for f in flags]].astype(bool).any(axis=1)

