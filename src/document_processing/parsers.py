import os
import re
import uuid
from pathlib import Path
from typing import List

import mammoth
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_equipment_type(file_path: Path, source_type: str) -> str:
    """Определяет тип оборудования по имени файла или источнику."""
    file_name_lower = file_path.name.lower()
    
    if 'astraregul' in file_name_lower or 'astra.regul' in file_name_lower:
        return "AstraRegul"
    elif 'r500s' in file_name_lower or 'safety' in file_name_lower:
        return "R500S"
    elif 'r500' in file_name_lower or 'regul' in file_name_lower:
        return "R500"
    elif 'ide' in file_name_lower:
        return "Astra.IDE"
    
    if source_type == 'r500s_docx': return "R500S"
    elif source_type in ['astraregul_docx', 'astraregul_html', 'metod_docx']: return "AstraRegul"
    elif source_type in ['r500_docx', 'r500_html']: return "R500"
    
    return "General"


def get_image_converter(images_out_dir: str):
    """Возвращает функцию-конвертер для Mammoth с привязкой к папке вывода."""
    os.makedirs(images_out_dir, exist_ok=True)
    
    def convert_image(image):
        ext = image.content_type.split("/")[-1]
        img_name = f"doc_img_{uuid.uuid4().hex[:8]}.{ext}"
        img_path = os.path.join(images_out_dir, img_name)
        with open(img_path, "wb") as f:
            with image.open() as image_bytes:
                f.write(image_bytes.read())
        return {"src": str(img_path)}
    
    return convert_image


def process_docx_file(
    file_path: Path, 
    source_type: str, 
    images_out_dir: str, 
    chunk_size: int = 2000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """Универсальный парсер DOCX файлов."""
    file_name = file_path.name
    documents = []
    equipment_type = detect_equipment_type(file_path, source_type)
    
    # Инициализация сплиттера
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "]
    )
    
    # Карты стилей
    style_maps = {
        'metod_docx': "p[style-name='Title'] => h1:fresh\np[style-name='Heading 1'] => h1:fresh\np[style-name='Heading 2'] => h2:fresh\np[style-name='Heading 3'] => h3:fresh\np[style-name='Heading 4'] => h4:fresh\np[style-name='Heading 5'] => h5:fresh\np[style-name='Heading 6'] => h6:fresh",
        'r500_docx': "p[style-name='ps2_Титул_Название продукта'] => h1:fresh\np[style-name='ps2_Заголовок 1'] => h1:fresh\np[style-name='ps2_Заголовок_Содержание'] => h1:fresh\np[style-name='ps2_Заголовок 2'] => h2:fresh\np[style-name='ps2_Заголовок 3'] => h3:fresh\np[style-name='ps2_Заголовок 4'] => h4:fresh\np[style-name='Title'] => h1:fresh\np[style-name='Heading 1'] => h1:fresh\np[style-name='Heading 2'] => h2:fresh\np[style-name='Heading 3'] => h3:fresh",
        'r500s_docx': "p[style-name='Title'] => h1:fresh\np[style-name='Heading 1'] => h1:fresh\np[style-name='Heading 2'] => h2:fresh\np[style-name='Heading 3'] => h3:fresh\np[style-name='Heading 4'] => h4:fresh",
        'astraregul_docx': "p[style-name='Title'] => h1:fresh\np[style-name='Heading 1'] => h1:fresh\np[style-name='Heading 2'] => h2:fresh\np[style-name='Heading 3'] => h3:fresh\np[style-name='Heading 4'] => h4:fresh\np[style-name='Heading 5'] => h5:fresh\np[style-name='Heading 6'] => h6:fresh",
    }
    custom_style_map = style_maps.get(source_type, style_maps['metod_docx'])
    
    try:
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(
                docx_file,
                style_map=custom_style_map,
                convert_image=mammoth.images.img_element(get_image_converter(images_out_dir))
            )
            html_content = result.value
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # --- Обработка таблиц в Markdown ---
            for table in soup.find_all('table'):
                parsed_table = []
                rowspans = {}
                for tr in table.find_all('tr'):
                    tds = tr.find_all(['td', 'th'])
                    if not tds: continue
                    row_data = []
                    col_idx, td_idx = 0, 0
                    while col_idx < len(tds) + len([k for k, v in rowspans.items() if v['span'] > 0]):
                        if col_idx in rowspans and rowspans[col_idx]['span'] > 0:
                            row_data.append(rowspans[col_idx]['text'])
                            rowspans[col_idx]['span'] -= 1
                            col_idx += 1
                        else:
                            if td_idx < len(tds):
                                td = tds[td_idx]
                                text = td.get_text(separator=" ", strip=True).replace('\n', ' ').replace('\r', '')
                                text = re.sub(r'(?i)(внимание|важно|примечание):', r'⚠️ **\1:**', text)
                                span = int(td.get('rowspan', 1))
                                if span > 1: rowspans[col_idx] = {'text': text, 'span': span - 1}
                                row_data.append(text)
                                td_idx += 1
                            else: row_data.append("")
                            col_idx += 1
                    if any(row_data): parsed_table.append("| " + " | ".join(row_data) + " |")
                    if len(parsed_table) == 1: parsed_table.append("|" + "|".join(["---"] * len(row_data)) + "|")
                if parsed_table: table.replace_with(soup.new_string(f"\n{chr(10).join(parsed_table)}\n"))
            
            # --- Обработка изображений ---
            for img in soup.find_all('img'):
                if src := img.get('src', ''):
                    img.replace_with(soup.new_string(f"\n![{img.get('alt', 'Изображение')}]({src})\n"))
            
            # --- Иерархический парсинг ---
            current_headers = {i: "" for i in range(1, 10)}
            current_chunk_content = []
            
            def save_chunk():
                if not current_chunk_content: return
                text_content = "\n".join(current_chunk_content)
                active_headers = [current_headers[i] for i in range(1, 10) if current_headers[i]]
                
                if active_headers and active_headers[0].lower() in ['содержание', 'оглавление', 'введение', 'аннотация']:
                    return
                
                breadcrumb_str = " > ".join(active_headers) if active_headers else "Документация"
                page_title = active_headers[-1] if active_headers else file_name.replace('.docx', '')
                text_content = re.sub(r'(?i)^(внимание|важно)!!!?', r'> ⚠️ **\1:**', text_content, flags=re.MULTILINE)
                
                base_meta = {
                    'source_file': file_name, 'equipment_type': equipment_type,
                    'breadcrumb_raw': breadcrumb_str, 'page_title': page_title,
                    'format': 'docx', 'source_type': source_type,
                    'has_table': "| --- |" in text_content
                }
                # Сохраняем только непустые метаданные
                base_meta = {k: v for k, v in base_meta.items() if v or isinstance(v, bool)}
                
                if len(text_content) > chunk_size:
                    for i, sub_text in enumerate(text_splitter.split_text(text_content)):
                        meta = base_meta.copy()
                        meta['chunk_part'] = f"{i+1}"
                        documents.append(Document(page_content=f"[ДОКУМЕНТ: {breadcrumb_str}]\n{sub_text}", metadata=meta))
                else:
                    documents.append(Document(page_content=f"[ДОКУМЕНТ: {breadcrumb_str}]\n{text_content}", metadata=base_meta))
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    save_chunk()
                    current_chunk_content = []
                    level = int(element.name[1])
                    current_headers[level] = element.get_text(strip=True)
                    for i in range(level + 1, 10): current_headers[i] = ""
                else:
                    if text := element.get_text(separator='\n', strip=True):
                        current_chunk_content.append(text)
            save_chunk()
            
    except Exception as e:
        print(f"❌ Ошибка парсинга DOCX {file_name}: {e}")
    
    return documents


def process_html_file(
    file_path: Path, 
    source_type: str, 
    base_url: str = "", 
    chunk_size: int = 2000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """Универсальный парсер для HTML файлов документации."""
    file_name = file_path.name
    documents = []
    equipment_type = detect_equipment_type(file_path, source_type)
    
    # Инициализация сплиттера
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "]
    )
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # --- Глубокая очистка от мусора (скрипты, стили, меню) ---
        for tag in soup(["script", "style", "meta", "noscript", "link", "head", "iframe", "svg"]):
            tag.decompose()
        for tag in soup(["nav", "footer", "header", "aside"]):
            tag.decompose()
            
        garbage_keywords = re.compile(r'(?i)menu|nav|header|footer|toc|sidebar|breadcrumbs')
        for tag in soup.find_all(attrs={"id": garbage_keywords}): tag.decompose()
        for tag in soup.find_all(attrs={"class": garbage_keywords}): tag.decompose()

        # --- Извлечение метаданных ---
        breadcrumbs = []
        bc_ul = soup.find('ul', class_='b-breadCrumbs__items')
        if bc_ul:
            for li in bc_ul.find_all('li'): breadcrumbs.append(li.get_text(strip=True))
        breadcrumb_str = " > ".join(breadcrumbs) if breadcrumbs else "Документация"
        
        h1_tag = soup.find('h1')
        page_title = h1_tag.get_text(strip=True) if h1_tag else file_name.replace('.htm', '').replace('.html', '')
        
        next_link_tag = soup.find('a', title='Следующая')
        next_file = next_link_tag['href'] if next_link_tag and next_link_tag.has_attr('href') else ""
        
        # --- Основной контент ---
        article = soup.find('article') or soup.find('div', class_='b-article__wrapper') or soup.body or soup
        if not article:
            return []
        
        # --- Обработка таблиц ---
        for table in article.find_all('table'):
            parsed_table = []
            rowspans = {}
            for tr in table.find_all('tr'):
                tds = tr.find_all(['td', 'th'])
                if not tds: continue
                row_data = []
                col_idx, td_idx = 0, 0
                while col_idx < len(tds) + len([k for k, v in rowspans.items() if v['span'] > 0]):
                    if col_idx in rowspans and rowspans[col_idx]['span'] > 0:
                        row_data.append(rowspans[col_idx]['text'])
                        rowspans[col_idx]['span'] -= 1
                        col_idx += 1
                    else:
                        if td_idx < len(tds):
                            td = tds[td_idx]
                            text = td.get_text(separator=" ", strip=True).replace('\n', ' ').replace('\r', '')
                            # Замена иконок-шрифтов на Markdown
                            text = text.replace('', 'ℹ️ **ИНФОРМАЦИЯ:**').replace('', '⚠️ **ВНИМАНИЕ:**').replace('', '⚠️ **ВНИМАНИЕ:**')
                            span = int(td.get('rowspan', 1))
                            if span > 1: rowspans[col_idx] = {'text': text, 'span': span - 1}
                            row_data.append(text)
                            td_idx += 1
                        else: row_data.append("")
                        col_idx += 1
                if any(row_data): parsed_table.append("| " + " | ".join(row_data) + " |")
                if len(parsed_table) == 1: parsed_table.append("|" + "|".join(["---"] * len(row_data)) + "|")
                
            if parsed_table:
                new_tag = soup.new_tag("div")
                new_tag.string = f"\n\n{chr(10).join(parsed_table)}\n\n"
                table.replace_with(new_tag)
        
        # Извлекаем финальный текст
        text_content = article.get_text(separator='\n', strip=True)
        text_content = text_content.replace('', '> ℹ️ **ИНФОРМАЦИЯ:**\n> ').replace('', '> ⚠️ **ВНИМАНИЕ:**\n> ').replace('', '> ⚠️ **ВНИМАНИЕ:**\n> ')
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        
        if len(text_content) < 20:
            return []
        
        nav_text = f"\n\n[Следующий раздел: {base_url}{next_file}]" if next_file and base_url else ""
        final_text = f"[РАЗДЕЛ: {breadcrumb_str}]\n[ЗАГОЛОВОК: {page_title}]\n{text_content}{nav_text}"
        
        base_meta = {
            'source_file': file_name,
            'equipment_type': equipment_type,
            'source_url': f"{base_url}{file_name}" if base_url else "",
            'page_title': page_title,
            'breadcrumb_raw': breadcrumb_str,
            'format': 'html',
            'source_type': source_type,
            'has_table': "| --- |" in final_text
        }
        # Убираем пустые значения
        base_meta = {k: v for k, v in base_meta.items() if v or isinstance(v, bool)}
        
        # Нарезка на чанки
        if len(final_text) > chunk_size:
            sub_chunks = text_splitter.split_text(final_text)
            for i, sub in enumerate(sub_chunks):
                sub_meta = base_meta.copy()
                sub_meta['chunk_part'] = f"{i+1}/{len(sub_chunks)}"
                
                # Добавляем контекст в начало каждого чанка, если его там нет
                content = f"[РАЗДЕЛ: {breadcrumb_str}]\n[ЗАГОЛОВОК: {page_title}]\n{sub}" if not sub.startswith("[РАЗДЕЛ:") else sub
                sub_meta['chunk_length'] = len(content)
                documents.append(Document(page_content=content, metadata=sub_meta))
        else:
            base_meta['chunk_length'] = len(final_text)
            documents.append(Document(page_content=final_text, metadata=base_meta))
            
    except Exception as e:
        print(f"❌ Ошибка парсинга HTML {file_name}: {e}")
    
    return documents