# Полный код server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import time
import uvicorn # Нужно для запуска внутри скрипта

app = FastAPI()

# Разрешаем CORS (чтобы сайт видел сервер)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

def get_article_full_text(url):
    try:
        # Притворяемся обычным браузером, чтобы нас не заблокировали
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Пытаемся найти основной текст
        content = soup.find('article') or soup.find('main') or soup.body
        
        if not content:
            return None

        text_parts = []
        # Собираем заголовки, параграфы и списки
        for tag in content.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            text_parts.append(tag.get_text().strip())
        
        return '\n\n'.join(filter(None, text_parts))
    except Exception as e:
        print(f"Ошибка скачивания: {e}")
        return None

def split_text(text, max_chars=4500):
    """Режет текст на куски, чтобы не сломать переводчик"""
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
    start_time = time.time()
    
    # 1. Скачиваем
    original_text = get_article_full_text(request.url)
    if not original_text:
        return {"error": "Не удалось скачать текст. Сайт защищен или недоступен."}

    # 2. Разбиваем
    chunks = split_text(original_text)
    translated_chunks = []
    translator = GoogleTranslator(source='auto', target='ru')

    # 3. Переводим
    try:
        for chunk in chunks:
            if chunk.strip():
                res = translator.translate(chunk)
                translated_chunks.append(res)
            else:
                translated_chunks.append("")
        
        full_translation = '\n\n'.join(translated_chunks)
        
        duration = round(time.time() - start_time, 2)
        word_count = len(full_translation.split())
        
        return {
            "original": original_text,
            "translated": full_translation,
            "stats": f"Переведено {word_count} слов за {duration} сек."
        }
        
    except Exception as e:
        return {"error": f"Ошибка перевода: {str(e)}"}

# --- ВОТ ЭТОТ КУСОК ЗАПУСКАЕТ СЕРВЕР ---
if __name__ == "__main__":
    print("Запускаем сервер... Не закрывай это окно!")
    # host="0.0.0.0" позволяет видеть сервер даже с телефона в локальной сети
    uvicorn.run(app, host="192.168.1.6", port=8001)




