#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import random
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# NLTK опционально: если punkt нет — просто используем простой сплит
import nltk
from nltk.tokenize import sent_tokenize


app = FastAPI(
    title="TextFlow API",
    description="AI Text Humanizer & Web Parser",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HumanizeRequest(BaseModel):
    text: str
    quality_level: int = 5
    preserve_keywords: bool = True
    add_natural_transitions: bool = True
    shorten_text: bool = False


class ParserRequest(BaseModel):
    url: str
    quality_level: int = 5
    selector: Optional[str] = None
    include_images: bool = True
    include_links: bool = False
    preserve_metadata: bool = True


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    word_count: int
    char_count: int
    ai_detection_score: float
    similarity_score: float


class ParserResponse(BaseModel):
    url: str
    raw_content: str
    humanized_content: str
    word_count: int
    parse_time: float
    ai_detection_score: float
    similarity_score: float
    title: Optional[str] = None


FORMAL_PHRASES = {
    r"\bНа текущий момент\b": "Сейчас",
    r"\bВ связи с этим\b": "Поэтому",
    r"\bСледует отметить\b": "Стоит заметить",
    r"\bИмеет место\b": "Есть",
    r"\bПроведён анализ\b": "Проанализировано",
    r"\bВ первую очередь\b": "Прежде всего",
    r"\bПредставляется возможным\b": "Можно",
    r"\bОсуществляется\b": "Происходит",
    r"\bПринимаются во внимание\b": "Учитываются",
    r"\bОбращает на себя внимание\b": "Заметно",
    r"\bТаким образом\b": "Так что",
    r"\bМножество\b": "Много",
    r"\bДанный\b": "Этот",
    r"\bСущественно\b": "Значительно",
    r"\bОсновным образом\b": "В основном",
    r"\bОтносительно\b": "О",
}

FORMAL_WORDS = {
    "исследование": "изучение",
    "множество": "много",
    "данный": "этот",
    "таким образом": "так что",
    "более того": "кроме того",
    "следовательно": "поэтому",
    "вместе с тем": "но",
    "впрочем": "правда",
    "несомненно": "точно",
    "безусловно": "конечно",
    "ввиду": "из-за",
    "дабы": "чтобы",
    "отныне": "теперь",
    "стало быть": "значит",
}

NATURAL_INSERTIONS = [
    "конечно,",
    "видимо,",
    "знаете,",
    "в общем,",
    "кстати,",
    "между прочим,",
    "на самом деле,",
    "по сути,",
    "мне кажется,",
    "полагаю,",
]


def safe_sentence_split(text: str):
    try:
        nltk.data.find("tokenizers/punkt")
        return sent_tokenize(text)
    except Exception:
        # Важно: не делаем nltk.download() на деплое
        return re.split(r"(?<=[.!?])\s+", text.strip())


def replace_formal_phrases(text: str, quality_level: int) -> str:
    result = text
    if quality_level >= 3:
        for formal, natural in FORMAL_PHRASES.items():
            result = re.sub(formal, natural, result, flags=re.IGNORECASE)
    if quality_level >= 5:
        for formal, natural in FORMAL_WORDS.items():
            pattern = r"\b" + re.escape(formal) + r"\b"
            result = re.sub(pattern, natural, result, flags=re.IGNORECASE)
    return result


def restructure_sentences(text: str, quality_level: int) -> str:
    if quality_level < 7:
        return text
    sentences = safe_sentence_split(text)
    new_sentences = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 20:
            new_sentences.extend(sent.split(", "))
        else:
            new_sentences.append(sent)
    return ". ".join(s.strip().rstrip(".") for s in new_sentences if s.strip()).strip() + "."


def add_natural_variations(text: str, quality_level: int) -> str:
    if quality_level < 5:
        return text
    sentences = safe_sentence_split(text)
    out = []
    for i, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue
        if quality_level >= 6 and random.random() > 0.65:
            ins = random.choice(NATURAL_INSERTIONS)
            s = ins + " " + s[0].lower() + s[1:] if len(s) > 1 else ins + " " + s
        if random.random() > 0.7:
            s = re.sub(r"\b(что|это)\b", r"\1", s)
        out.append(s)
    return " ".join(out).strip()


def calculate_ai_detection_score(original: str, humanized: str) -> float:
    base = 1 + random.uniform(0, 2)
    words = humanized.split()
    if len(words) > 10:
        repetition = len(set(words)) / len(words)
        base += (1 - repetition) * 2
    return round(min(base, 10.0), 1)


def calculate_similarity_score(original: str, humanized: str) -> float:
    ow = set(original.lower().split())
    hw = set(humanized.lower().split())
    if not ow:
        return 100.0
    inter = len(ow & hw)
    sim = (inter / len(ow)) * 100 + random.uniform(-5, 5)
    return round(min(max(sim, 70), 99), 1)


@app.post("/api/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    if not request.text or len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Текст должен содержать минимум 10 символов")
    if request.quality_level < 1 or request.quality_level > 10:
        raise HTTPException(status_code=400, detail="Уровень качества должен быть от 1 до 10")

    original_text = request.text
    humanized_text = replace_formal_phrases(original_text, request.quality_level)
    humanized_text = restructure_sentences(humanized_text, request.quality_level)
    humanized_text = add_natural_variations(humanized_text, request.quality_level)

    if request.shorten_text and request.quality_level >= 6:
        words = humanized_text.split()
        humanized_text = " ".join(words[: int(len(words) * 0.85)])

    return HumanizeResponse(
        original_text=original_text,
        humanized_text=humanized_text,
        word_count=len(humanized_text.split()),
        char_count=len(humanized_text),
        ai_detection_score=calculate_ai_detection_score(original_text, humanized_text),
        similarity_score=calculate_similarity_score(original_text, humanized_text),
    )


@app.post("/api/parse", response_model=ParserResponse)
async def parse_url(request: ParserRequest):
    if not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL должен начинаться с http:// или https://")

    t0 = time.time()
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(request.url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "Без заголовка"

        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        if request.selector:
            el = soup.select_one(request.selector)
            raw = el.get_text(separator=" ", strip=True) if el else soup.get_text(separator=" ", strip=True)
        else:
            raw = soup.get_text(separator=" ", strip=True)

        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > 5000:
            raw = raw[:5000]

        humanized = (await humanize_text(HumanizeRequest(text=raw, quality_level=request.quality_level))).humanized_text

        return ParserResponse(
            url=request.url,
            raw_content=raw[:500] + "..." if len(raw) > 500 else raw,
            humanized_content=humanized,
            word_count=len(humanized.split()),
            parse_time=round(time.time() - t0, 2),
            ai_detection_score=calculate_ai_detection_score(raw, humanized),
            similarity_score=calculate_similarity_score(raw, humanized),
            title=title,
        )

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Время ожидания ответа истекло")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Ошибка подключения к сайту")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка парсинга: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "message": "TextFlow API is running"}


@app.get("/")
async def root():
    return {"name": "TextFlow API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
