from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import time

app = FastAPI()

# CORS: Разрешаем всем стучаться к нам
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить любые источники (GitHub Pages, локальный файл и т.д.)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить любые методы (GET, POST)
    allow_headers=["*"],  # Разрешить любые заголовки
)

class LinkRequest(BaseModel):
    url: str

def get_article_full_text(url):
    try:
        # Притворяемся браузером Chrome
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15) # Увеличил таймаут до 15 сек
        
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Улучшенная логика поиска контента
        content = soup.find('article') 
        if not content:
            content = soup.find('main')
        if not content:
            # Если не нашли явных тегов, ищем самый большой блок с текстом
            content = max(soup.find_all('div'), key=lambda tag: len(tag.get_text()), default=soup.body)
        
        if not content:
            return None

        text_parts = []
        for tag in content.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            text = tag.get_text().strip()
            if len(text) > 20: # Отсеиваем мусор (слишком короткие строки)
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return None

def split_text(text, max_chars=4500):
    chunks = []
    while len(text) > max_chars:
        split_index = text.rfind('.', 0, max_chars)
        if split_index == -1:
            split_index = text.rfind('\n', 0, max_chars)
        if split_index == -1:
            split_index = max_chars
        chunks.append(text[:split_index+1])
        text = text[split_index+1:]
    chunks.append(text)
    return chunks

@app.post("/translate")
async def translate_article(request: LinkRequest):
    # Проверка: если пользователь забыл http
    if not request.url.startswith('http'):
        return {"error": "Ссылка должна начинаться с http:// или https://"}

    start_time = time.time()
    
    # 1. Скачиваем
    original_text = get_article_full_text(request.url)
    if not original_text:
        return {"error": "Не удалось скачать текст. Сайт защищен или недоступен."}

    # 2. Режем и переводим
    chunks = split_text(original_text)
    translated_chunks = []
    translator = GoogleTranslator(source='auto', target='ru')

    try:
        for chunk in chunks:
            if chunk.strip():
                res = translator.translate(chunk)
                translated_chunks.append(res)
        
        full_translation = '\n\n'.join(translated_chunks)
        duration = round(time.time() - start_time, 2)
        word_count = len(full_translation.split())
        
        return {
            "original": original_text,
            "translated": full_translation,
            "stats": f"Переведено {word_count} слов за {duration} сек."
        }
        
    except Exception as e:
        return {"error": f"Сбой переводчика: {str(e)}"}

# Этот блок нужен ТОЛЬКО для локального запуска на компе. 
# Render его игнорирует, так как использует свою команду запуска.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
