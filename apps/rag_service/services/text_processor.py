from typing import List, Set
from .clients import get_spacy_nlp, get_yake_extractor, get_keybert_model

KEEP_LABELS: Set[str] = {
    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "NORP", "FAC",
    "LOC", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
    "MONEY", "PERCENT"
}

class EnhancedTextProcessor:
    def extract_entities(self, text: str) -> List[str]:
        nlp = get_spacy_nlp()
        doc = nlp(text[:1_000_000])               # safety clip
        return list({ent.text for ent in doc.ents if ent.label_ in KEEP_LABELS})

    def extract_keywords(self, text: str) -> List[str]:
        keywords = set()

        yake_kws = [kw for kw, _ in get_yake_extractor().extract_keywords(text)]
        keywords.update(yake_kws[:5])

        try:
            keybert_kws = get_keybert_model().extract_keywords(
                text, keyphrase_ngram_range=(1, 3),
                stop_words="english", top_n=5
            )
            keywords.update([kw for kw, _ in keybert_kws])
        except Exception:
            pass

        return list(keywords)[:10]

processor = EnhancedTextProcessor()