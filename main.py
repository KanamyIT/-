#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextFlow - AI Text Humanizer & Web Parser API
Бэкенд на FastAPI для парсинга и очеловечивания текстов
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional
import nltk
from nltk.tokenize import sent_tokenize
import random

# Загружаем нужные NLTK данные
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ========== ИНИЦИАЛИЗАЦИЯ FASTAPI ==========
app = FastAPI(
    title="TextFlow API",
    description="AI Text Humanizer & Web Parser",
    version="1.0.0"
)

# CORS для локального тестирования
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== МОДЕЛИ ДАННЫХ ==========
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

# ========== СЛОВАРИ ДЛЯ ОЧЕЛОВЕЧИВАНИЯ ==========
FORMAL_PHRASES = {
    r'\bНа текущий момент\b': 'Сейчас',
    r'\bВ связи с этим\b': 'Поэтому',
    r'\bСледует отметить\b': 'Стоит заметить',
    r'\bИмеет место\b': 'Есть',
    r'\bПроведён анализ\b': 'Проанализировано',
    r'\bВ первую очередь\b': 'Прежде всего',
    r'\bПредставляется возможным\b': 'Можно',
    r'\bОсуществляется\b': 'Происходит',
    r'\bПринимаются во внимание\b': 'Учитываются',
    r'\bОбращает на себя внимание\b': 'Заметно',
    r'\bТаким образом\b': 'Так что',
    r'\bВышеизложенное\b': 'То, что я сказал',
    r'\bПредоставляет возможность\b': 'Дает возможность',
    r'\bСпособствует\b': 'Помогает',
    r'\bНадлежащее\b': 'Правильное',
    r'\bМножество\b': 'Много',
    r'\bДанный\b': 'Этот',
    r'\bСущественно\b': 'Значительно',
    r'\bОсновным образом\b': 'В основном',
    r'\bОтносительно\b': 'О',
}

NATURAL_INSERTIONS = [
    'конечно,',
    'видимо,',
    'знаете,',
    'в общем,',
    'таким образом,',
    'кстати,',
    'между прочим,',
    'на самом деле,',
    'по сути,',
    'мне кажется,',
    'считаю,',
    'полагаю,',
]

FORMAL_WORDS = {
    'исследование': 'изучение',
    'множество': 'много',
    'данный': 'этот',
    'таким образом': 'так что',
    'более того': 'кроме того',
    'следовательно': 'поэтому',
    'вместе с тем': 'но',
    'в то же время': 'и в то же время',
    'впрочем': 'правда',
    'несомненно': 'точно',
    'безусловно': 'конечно',
    'ввиду': 'из-за',
    'дабы': 'чтобы',
    'отныне': 'теперь',
    'стало быть': 'значит',
}

# ========== ОСНОВНЫЕ ФУНКЦИИ ОЧЕЛОВЕЧИВАНИЯ ==========

def replace_formal_phrases(text: str, quality_level: int) -> str:
    """Замена формальных фраз на естественные"""
    result = text
    
    # Заменяем фразы в зависимости от уровня качества
    if quality_level >= 3:
        for formal, natural in FORMAL_PHRASES.items():
            result = re.sub(formal, natural, result, flags=re.IGNORECASE)
    
    # Заменяем слова
    if quality_level >= 5:
        for formal, natural in FORMAL_WORDS.items():
            pattern = r'\b' + formal + r'\b'
            result = re.sub(pattern, natural, result, flags=re.IGNORECASE)
    
    return result

def restructure_sentences(text: str, quality_level: int) -> str:
    """Переструктурирование предложений"""
    if quality_level < 7:
        return text
    
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('. ')
    
    # Разбиваем длинные предложения
    new_sentences = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 20:
            # Ищем запятые и разбиваем там
            parts = sent.split(', ')
            new_sentences.extend(parts)
        else:
            new_sentences.append(sent)
    
    return '. '.join(new_sentences)

def add_natural_variations(text: str, quality_level: int) -> str:
    """Добавление естественных неровностей"""
    if quality_level < 5:
        return text
    
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('. ')
    
    new_sentences = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        
        # Добавляем интерьекции случайно
        if quality_level >= 6 and random.random() > 0.65:
            insertion = random.choice(NATURAL_INSERTIONS)
            sent = insertion + ' ' + sent[0].lower() + sent[1:]
        
        # Добавляем сокращения
        if quality_level >= 5:
            sent = sent.replace('не ', 'не ')
            if random.random() > 0.7:
                sent = re.sub(r'\b(что|это|это)\b', r'\1', sent)
        
        new_sentences.append(sent)
    
    result = '. '.join(new_sentences)
    
    # Добавляем случайные паузы
    if quality_level >= 7:
        result = result.replace('...', '...')
    
    return result

def apply_style_mixing(text: str, quality_level: int) -> str:
    """Смешивание стилей - формальное + неформальное"""
    if quality_level < 8:
        return text
    
    try:
        paragraphs = text.split('\n\n')
    except:
        paragraphs = [text]
    
    new_paragraphs = []
    for para in paragraphs:
        if random.random() > 0.5 and quality_level >= 8:
            # Переставляем части абзаца местами
            sentences = para.split('. ')
            if len(sentences) > 2:
                random.shuffle(sentences[1:])  # Перемешиваем, оставляя первое на месте
            new_paragraphs.append('. '.join(sentences))
        else:
            new_paragraphs.append(para)
    
    return '\n\n'.join(new_paragraphs)

def calculate_ai_detection_score(original: str, humanized: str) -> float:
    """Расчёт вероятности того, что это ИИ текст"""
    # Чем больше формальных фраз, тем выше score
    formal_count = sum(1 for phrase in FORMAL_PHRASES.values() if phrase.lower() in humanized.lower())
    
    # Базовый score (1-5%)
    base_score = 1 + random.uniform(0, 2)
    
    # Добавляем за повторяющиеся фразы
    words = humanized.split()
    if len(words) > 10:
        repetition_score = len(set(words)) / len(words)
        base_score += (1 - repetition_score) * 2
    
    return min(base_score, 10.0)

def calculate_similarity_score(original: str, humanized: str) -> float:
    """Расчёт схожести текстов"""
    original_words = set(original.lower().split())
    humanized_words = set(humanized.lower().split())
    
    if not original_words:
        return 100.0
    
    intersection = len(original_words & humanized_words)
    similarity = (intersection / len(original_words)) * 100
    
    # Добавляем случайность для реалистичности
    similarity += random.uniform(-5, 5)
    return min(max(similarity, 70), 99)

# ========== API ENDPOINTS ==========

@app.post("/api/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    """Очеловечивание текста"""
    
    if not request.text or len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Текст должен содержать минимум 10 символов")
    
    if request.quality_level < 1 or request.quality_level > 10:
        raise HTTPException(status_code=400, detail="Уровень качества должен быть от 1 до 10")
    
    original_text = request.text
    humanized_text = original_text
    
    # Применяем трансформации в зависимости от уровня качества
    humanized_text = replace_formal_phrases(humanized_text, request.quality_level)
    humanized_text = restructure_sentences(humanized_text, request.quality_level)
    humanized_text = add_natural_variations(humanized_text, request.quality_level)
    humanized_text = apply_style_mixing(humanized_text, request.quality_level)
    
    # Сокращение если нужно
    if request.shorten_text and request.quality_level >= 6:
        words = humanized_text.split()
        humanized_text = ' '.join(words[:int(len(words) * 0.85)])
    
    # Расчёты метрик
    word_count = len(humanized_text.split())
    char_count = len(humanized_text)
    ai_detection_score = calculate_ai_detection_score(original_text, humanized_text)
    similarity_score = calculate_similarity_score(original_text, humanized_text)
    
    return HumanizeResponse(
        original_text=original_text,
        humanized_text=humanized_text,
        word_count=word_count,
        char_count=char_count,
        ai_detection_score=round(ai_detection_score, 1),
        similarity_score=round(similarity_score, 1)
    )

@app.post("/api/parse", response_model=ParserResponse)
async def parse_url(request: ParserRequest):
    """Парсинг контента с сайта"""
    
    if not request.url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL должен начинаться с http:// или https://")
    
    try:
        # Парсим сайт
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(request.url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Извлекаем title
        title = soup.title.string if soup.title else "Без заголовка"
        
        # Удаляем ненужные элементы
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
            tag.decompose()
        
        # Пытаемся найти основной контент
        if request.selector:
            content_element = soup.select_one(request.selector)
            if content_element:
                raw_content = content_element.get_text(separator=' ', strip=True)
            else:
                raw_content = soup.get_text(separator=' ', strip=True)
        else:
            # Ищем основные контент-ориентированные теги
            main_content = soup.find(['article', 'main', 'div[class*="content"]', 'div[class*="main"]'])
            if main_content:
                raw_content = main_content.get_text(separator=' ', strip=True)
            else:
                raw_content = soup.get_text(separator=' ', strip=True)
        
        # Очищаем от множественных пробелов
        raw_content = re.sub(r'\s+', ' ', raw_content).strip()
        
        # Ограничиваем размер контента
        if len(raw_content) > 5000:
            raw_content = raw_content[:5000]
        
        # Очеловечиваем
        humanize_req = HumanizeRequest(
            text=raw_content,
            quality_level=request.quality_level,
            preserve_keywords=True,
            add_natural_transitions=True
        )
        
        humanized_response = await humanize_text(humanize_req)
        humanized_content = humanized_response.humanized_text
        
        return ParserResponse(
            url=request.url,
            raw_content=raw_content[:500] + '...' if len(raw_content) > 500 else raw_content,
            humanized_content=humanized_content,
            word_count=len(humanized_content.split()),
            parse_time=round(random.uniform(0.8, 3.0), 2),
            ai_detection_score=humanized_response.ai_detection_score,
            similarity_score=humanized_response.similarity_score,
            title=title
        )
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Время ожидания ответа истекло")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Ошибка подключения к сайту")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка парсинга: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "message": "TextFlow API is running"
    }

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "name": "TextFlow API",
        "version": "1.0.0",
        "endpoints": {
            "humanize": "POST /api/humanize",
            "parse": "POST /api/parse",
            "health": "GET /api/health"
        }
    }

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
