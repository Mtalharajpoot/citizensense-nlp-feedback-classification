"""Shared logic for CitizenSense: model loading, keywords, summary, department routing."""
import os
import re
import joblib

BASE = os.environ.get("NLP_TASK_DIR", "/content/drive/MyDrive/nlp_task_4")
MODEL_DIR = os.path.join(BASE, "models")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_state = {}


def _load():
    """Load models once (sentence-transformer + 3 classifiers)."""
    if _state:
        return _state
    from sentence_transformers import SentenceTransformer
    _state["st"] = SentenceTransformer(EMBED_MODEL)
    _state["models"] = {
        t: joblib.load(os.path.join(MODEL_DIR, f"{t}_model.pkl"))
        for t in ["category", "sentiment", "priority"]
    }
    return _state


def encode(texts):
    st = _load()["st"]
    return st.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def predict(target, text):
    """Category and sentiment use embeddings + LogReg; priority uses char TF-IDF + LogReg."""
    m = _load()["models"][target]
    X = [text] if target == "priority" else encode([text])
    p = m.predict_proba(X)[0]
    return m.classes_[p.argmax()], float(p.max())


STOP = set("""a an the is are was were be been being am of in on at to for from by with and or but not no nor so than too
very this that these those it its our your their my we you they he she i me us them has have had do does did will would
can could should may might shall please kindly also just still yet again every each any some there here near about after
before during over under into out up down off as if when while where which who what how why all get gets got given
hai hain ho hota hoti hote ka ki ke ko se mein par aur ya bhi to ye wo yeh woh hamare hamari hamara apne apna kar kya
kyun kyunke jo jab tak nhi nahi na bohat bahut hum tum aap ab abhi ek hay ha raha rahi rahe gaya gayi gaye dein den
area""".split())


def keywords(text, top_n=4):
    """Rank 1-2 word candidates by embedding similarity to the full text (KeyBERT-style)."""
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    cands = set()
    for n in (1, 2):
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if any(len(w) < 3 for w in g) or g[0] in STOP or g[-1] in STOP:
                continue
            cands.add(" ".join(g))
    if not cands:
        return []
    cands = sorted(cands)
    sims = (encode(cands) @ encode([text]).T).ravel()
    chosen, covered = [], set()
    for idx in sims.argsort()[::-1]:
        c = cands[idx]
        ws = set(c.split())
        if ws <= covered:
            continue
        chosen.append(c)
        covered |= ws
        if len(chosen) == top_n:
            break
    return chosen


# Example mapping only, not an official government routing table.
DEPT = {
    "Roads / Infrastructure": ("Municipal / Infrastructure (Works and Services)",
                               "Inspect and repair the reported road or infrastructure issue"),
    "Water Supply": ("Water and Sanitation Authority / Public Health Engineering",
                     "Check the water line or supply schedule and restore service"),
    "Electricity": ("Electricity Distribution Company / Power Department",
                    "Send a technical team to inspect the fault"),
    "Sanitation": ("Municipal Sanitation / Solid Waste Management",
                   "Schedule cleaning, garbage collection or sewer clearing"),
    "Health": ("District Health Authority / Health Department",
               "Review the service problem with the health facility"),
    "Traffic": ("Traffic Police / Transport Department",
                "Review signals, traffic management or enforcement at the location"),
}

# Simple heuristic (not a language-detection model).
ROMAN_HINTS = {"hai", "hain", "nahi", "nhi", "bohat", "aur", "hamare", "hamari", "mein", "bijli", "pani",
               "sarak", "gali", "kachra", "safai", "ganda", "nalkay", "kharab", "mushkil", "dawai",
               "mohalle", "ilaqe", "raha", "rahi", "aya", "ha"}


def analyze_feedback(text, review_threshold=0.45):
    text = re.sub(r"\s+", " ", text).strip()
    cat, cat_c = predict("category", text)
    sent, sent_c = predict("sentiment", text)
    prio, prio_c = predict("priority", text)
    kws = keywords(text)
    dept, action = DEPT[cat]
    kw_txt = ", ".join(kws) if kws else "n/a"
    if sent == "Positive":
        summary = f"Citizen shares positive feedback about {cat} services ({kw_txt})."
        action = "Acknowledge the feedback and share it with the team"
    elif sent == "Neutral":
        summary = f"Citizen shares a neutral or mixed comment or request about {cat} ({kw_txt})."
    else:
        summary = f"Citizen reports a {cat} problem ({kw_txt})."
        if prio in ("High", "Urgent"):
            action = "Priority handling: " + action
    flags = []
    if cat_c < review_threshold:
        flags.append("low category confidence")
    if prio == "Urgent":
        flags.append("urgent priority")
    if set(re.findall(r"[a-z]+", text.lower())) & ROMAN_HINTS:
        flags.append("Roman Urdu input (model trained mostly on English)")
    return {"text": text,
            "category": cat, "category_score": round(cat_c, 2),
            "sentiment": sent, "sentiment_score": round(sent_c, 2),
            "priority": prio, "priority_score": round(prio_c, 2),
            "keywords": kws, "suggested_department": dept, "summary": summary,
            "suggested_action": action,
            "needs_manual_review": bool(flags), "review_reason": ", ".join(flags)}


def format_card(r):
    review = ("YES (" + r["review_reason"] + ")") if r["needs_manual_review"] else "no"
    return "\n".join([
        "-" * 70,
        f"Input     : {r['text']}",
        f"Category  : {r['category']} ({r['category_score']})",
        f"Sentiment : {r['sentiment']} ({r['sentiment_score']})",
        f"Priority  : {r['priority']} ({r['priority_score']})  [indicative only]",
        f"Keywords  : {', '.join(r['keywords'])}",
        f"Department: {r['suggested_department']}",
        f"Summary   : {r['summary']}",
        f"Action    : {r['suggested_action']}",
        f"Review    : {review}",
        "-" * 70,
    ])
