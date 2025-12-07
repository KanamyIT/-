from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup, NavigableString
from deep_translator import GoogleTranslator
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

def translate_html_content(soup, translator):
    """
    Мощный рекурсивный переводчик. Заходит в каждый тег.
    """
    # Ищем основной контент (чтобы не переводить меню и футер)
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.find('div', class_='post') or soup.body
    
    if not content:
        return "<p>Не удалось найти контент :(</p>"

    # Убираем скрипты и стили, чтобы они не сломались
    for script in content(["script", "style", "iframe", "noscript"]):
        script.decompose() # Удаляем их из дерева

    # Проходим по всем элементам, у которых есть текст
    # Используем find_all(text=True) для поиска именно текстовых узлов
    text_nodes = content.find_all(text=True)

    for node in text_nodes:
        # Пропускаем пустые строки и системные символы
        if isinstance(node, NavigableString) and len(node.strip()) > 3:
            # Пропускаем текст, если он внутри ссылок (иногда это ломает верстку) или кода
            parent_name = node.parent.name
            if parent_name in ['script', 'style', 'code', 'pre']:
                continue
                
            original_text = str(node)
            try:
                # Переводим кусочек
                translated_text = translator.translate(original_text)
                # Заменяем в HTML
                node.replace_with(translated_text)
            except:
                pass # Если ошибка, оставляем как было

    # Обработка картинок (чтобы они были красивыми)
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        # Фикс для ленивой загрузки (data-src -> src)
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']

    # Возвращаем только HTML внутри контента
    # prettify() делает код чистым
    return content.prettify()

@app.post("/translate")
async def translate_article(request: LinkRequest):
    start_time = time.time()
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.4896.127 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=25) # Таймаут побольше
        
        soup = BeautifulSoup(response.text, 'html.parser') # Используем lxml, если установлен, иначе html.parser
        translator = GoogleTranslator(source='auto', target='ru')

        final_html = translate_html_content(soup, translator)
        
        duration = round(time.time() - start_time, 2)
        
        return {
            "translated_html": final_html,
            "stats": f"Готово за {duration} сек."
        }
        
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
