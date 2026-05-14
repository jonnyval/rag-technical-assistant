import os
import re
import uuid
import logging  # <--- ДОБАВЛЕН ИМПОРТ
from pathlib import Path
from typing import List

import mammoth
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from src.document_processing.metadata_extractor import extract_smart_metadata

# === 🌟 ИНИЦИАЛИЗАЦИЯ ЛОГГЕРА ===
logger = logging.getLogger(__name__)

def detect_equipment_type(file_path: Path, source_type: str) -> str:
    """Определяет тип оборудования по имени файла и типу источника документации."""

    file_name_lower = file_path.name.lower()
    if 'astraregul' in file_name_lower or 'astra.regul' in file_name_lower: return "AstraRegul"
    elif 'r500s' in file_name_lower or 'safety' in file_name_lower: return "R500S"
    elif 'r500' in file_name_lower or 'regul' in file_name_lower: return "R500"
    elif 'ide' in file_name_lower: return "Astra.IDE"
    
    if source_type == 'r500s_docx': return "R500S"
    elif source_type in ['astraregul_docx', 'astraregul_html', 'metod_docx']: return "AstraRegul"
    elif source_type in ['r500_docx', 'r500_html']: return "R500"
    return "General"

def get_image_converter(images_out_dir: str):
    """Создает callback для Mammoth, который сохраняет картинки DOCX на диск."""

    os.makedirs(images_out_dir, exist_ok=True)
    def convert_image(image):
        """Сохраняет одно встроенное изображение DOCX и возвращает путь для HTML."""

        ext = image.content_type.split("/")[-1]
        img_name = f"doc_img_{uuid.uuid4().hex[:8]}.{ext}"
        img_path = os.path.join(images_out_dir, img_name)
        with open(img_path, "wb") as f:
            with image.open() as image_bytes:
                f.write(image_bytes.read())
        return {"src": str(img_path)}
    return convert_image

# ==========================================
# ТРЕК 1: DOCX (Семантические Разделы + Overlap)
# ==========================================
def process_docx_file(
    file_path: Path, 
    source_type: str, 
    images_out_dir: str
) -> List[Document]:
    """Парсит DOCX в семантические parent-документы с заголовками и metadata."""

    file_name = file_path.name
    documents = []
    equipment_type = detect_equipment_type(file_path, source_type)
    
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
            
            # Таблицы в Markdown
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
                
                if parsed_table:
                    new_tag = soup.new_tag("div")
                    new_tag.string = f"\n\n{chr(10).join(parsed_table)}\n\n"
                    table.replace_with(new_tag)
            
            # Списки в Markdown
            for ul in soup.find_all('ul'):
                for li in ul.find_all('li', recursive=False): li.insert_before(soup.new_string("- "))
            for ol in soup.find_all('ol'):
                for i, li in enumerate(ol.find_all('li', recursive=False)): li.insert_before(soup.new_string(f"{i+1}. "))
            
            current_headers = {i: "" for i in range(1, 10)}
            current_chunk_content = []
            chunk_counter = 0 # <--- 🌟 ДОБАВЛЕН СЧЕТЧИК
            
            def save_chunk():
                nonlocal current_chunk_content, chunk_counter # <--- 🌟 ДОБАВЛЕН СЧЕТЧИК
                if not current_chunk_content: return
                
                text_content = "\n".join(current_chunk_content)
                active_headers = [current_headers[i] for i in range(1, 10) if current_headers[i]]
                
                if active_headers and active_headers[0].lower() in ['содержание', 'оглавление', 'введение']:
                    current_chunk_content = []
                    return
                
                breadcrumb_str = " > ".join(active_headers) if active_headers else "Документация"
                page_title = active_headers[-1] if active_headers else file_name.replace('.docx', '')
                text_content = re.sub(r'(?i)^(внимание|важно)!!!?', r'> ⚠️ **\1:**', text_content, flags=re.MULTILINE)
                
                meta = {
                    'source_file': file_name, 
                    'equipment_type': equipment_type,
                    'breadcrumb_raw': breadcrumb_str, 
                    'page_title': page_title,
                    'format': 'docx', 
                    'source_type': source_type
                }
                
                final_text = f"[РАЗДЕЛ: {breadcrumb_str}]\n[ЗАГОЛОВОК: {page_title}]\n\n{text_content}"

                # === 🌟 "РЕНТГЕН" В ЛОГИ ===
                chunk_counter += 1
                logger.info(f"⏳ [Qwen] Файл: {file_name} | Чанк {chunk_counter} | {page_title[:40]}...")
                # ============================

                # === УМНОЕ ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ ===
                smart_meta = extract_smart_metadata(final_text)
                

                # Добавляем в текст для индексации
                if smart_meta.get("smart_questions"):
                    final_text += f"\n\n[ПОТЕНЦИАЛЬНЫЕ ВОПРОСЫ: {smart_meta['smart_questions']}]"
                if smart_meta.get("smart_keywords"):
                    final_text += f"\n[КЛЮЧЕВЫЕ СЛОВА: {smart_meta['smart_keywords']}]"

                # Добавляем в метаданные
                if smart_meta.get("smart_keywords"):
                    meta["keywords"] = smart_meta["smart_keywords"]
                if smart_meta.get("smart_questions"):
                    meta["generated_questions"] = smart_meta["smart_questions"]

                documents.append(Document(page_content=final_text, metadata=meta))
                
                # Overlap logic
                last_paragraph = current_chunk_content[-1] if current_chunk_content else ""
                if 20 < len(last_paragraph) < 500: 
                    current_chunk_content = [last_paragraph]
                else: 
                    current_chunk_content = []

            # Основной цикл обработки элементов
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'table']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    save_chunk()
                    level = int(element.name[1])
                    current_headers[level] = element.get_text(strip=True)
                    for i in range(level + 1, 10): current_headers[i] = ""
                else:
                    if text := element.get_text(separator='\n', strip=True):
                        current_chunk_content.append(text)
            
            save_chunk() # Финальный чанк
            
    except Exception as e:
        logger.error(f"❌ Ошибка парсинга DOCX {file_name}: {e}") # <--- ИСПОЛЬЗУЕМ ЛОГГЕР
    return documents

# ==========================================
# ТРЕК 2: HTML
# ==========================================
def detect_library_type(text: str, filename: str) -> str:
    """Определяет библиотеку/раздел HTML-документа по хлебным крошкам и имени файла."""

    filename_lower = filename.lower()
    text_sample = text[:2000].upper()
    if 'pstechmt' in filename_lower or 'МЕТАЛЛУРГИИ' in text_sample: return 'PsTechMT'
    elif 'pstechee' in filename_lower or 'ЭЛЕКТРОЭНЕРГЕТИК' in text_sample: return 'PsTechEE'
    elif 'psbase' in filename_lower or 'БАЗОВЫХ АЛГОРИТМОВ' in text_sample: return 'PsBase'
    elif 'psdiagn' in filename_lower or 'ДИАГНОСТИК' in text_sample: return 'PsDiagn'
    elif 'pssis' in filename_lower or 'PSSIS' in text_sample: return 'PSSIS'
    elif 'pstechog' in filename_lower or 'НЕФТЕГАЗОВ' in text_sample: return 'PsTechOG'
    return 'Универсальная'

def extract_function_block_name(text: str) -> str:
    """Извлекает имена функциональных блоков из текста или таблиц документации."""

    found_names = set()
    for pattern in [r'(FB_[A-Z0-9_]+)', r'\|\s*([A-Z0-9_]+)\s*\|', r'([A-Z0-9_]+)\s*\|\s*[А-Я]']:
        for fb_match in re.finditer(pattern, text):
            name = fb_match.group(1).strip()
            if len(name) >= 3 and name.isupper(): found_names.add(name)
    return ", ".join(sorted(found_names)) if found_names else ""

def process_html_file(file_path: Path, source_type: str, base_url: str = "") -> List[Document]:
    """Парсит HTML-страницу документации в parent-документ с техническими metadata."""

    file_name = file_path.name
    equipment_type = detect_equipment_type(file_path, source_type)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        breadcrumbs = []
        if bc_ul := soup.find('ul', class_='b-breadCrumbs__items'):
            for li in bc_ul.find_all('li'): breadcrumbs.append(li.get_text(strip=True))
        breadcrumb_str = " > ".join(breadcrumbs) if breadcrumbs else "Документация"
        
        library_key = detect_library_type(breadcrumb_str, file_name)
        fb_name = extract_function_block_name(breadcrumb_str)
        
        h1_tag = soup.find('h1')
        page_title = h1_tag.get_text(strip=True) if h1_tag else file_name.replace('.htm', '')

        release_version = "Неизвестно"
        for strong_tag in soup.find_all('strong'):
            text = strong_tag.get_text(strip=True)
            if "Релиз" in text or "Release" in text:
                release_version = text.replace('Релиз', '').strip()
                break

        next_link_tag = soup.find('a', title='Следующая')
        next_file = next_link_tag['href'] if next_link_tag and next_link_tag.has_attr('href') else ""
        
        article = soup.find('article') or soup.find('div', class_='b-article__wrapper') or soup.body or soup
        if not article: return []
            
        for hidden in article.find_all(['nav', 'script', 'style', 'noscript', 'footer', 'aside']): hidden.decompose()
        garbage_keywords = re.compile(r'(?i)menu|nav|header|footer|toc|sidebar')
        for tag in article.find_all(attrs={"id": garbage_keywords}): tag.decompose()

        for ul in article.find_all('ul'):
            for li in ul.find_all('li', recursive=False): li.insert_before(soup.new_string("- "))
        for ol in article.find_all('ol'):
            for i, li in enumerate(ol.find_all('li', recursive=False)): li.insert_before(soup.new_string(f"{i+1}. "))

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
                            text = text.replace('', 'ℹ️ **ИНФОРМАЦИЯ:**').replace('', '⚠️ **ВНИМАНИЕ:**')
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

        text_content = article.get_text(separator='\n', strip=True)
        text_content = text_content.replace('', '> ℹ️ **ИНФОРМАЦИЯ:**\n> ').replace('', '> ⚠️ **ВНИМАНИЕ:**\n> ')
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)

        if len(text_content) < 20: return []
        nav_text = f"\n\n[Следующий раздел: {base_url}{next_file}]" if next_file and base_url else ""
        
        final_text = (
            f"[ОБОРУДОВАНИЕ: {equipment_type}]\n"
            f"[БИБЛИОТЕКА: {library_key}]\n"
            f"[РЕЛИЗ: {release_version}]\n"
            f"[РАЗДЕЛ: {breadcrumb_str}]\n"
            f"[ЗАГОЛОВОК: {page_title}]\n\n"
            f"{text_content}{nav_text}"
        )

        # === 🌟 "РЕНТГЕН" В ЛОГИ ===
        logger.info(f"⏳ [Qwen] Файл: {file_name} | HTML Целиком | {page_title[:40]}...")
        # ============================

        # === УМНОЕ ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ ===
        smart_meta = extract_smart_metadata(final_text)

        if smart_meta.get("smart_questions"):
            final_text += f"\n\n[ПОТЕНЦИАЛЬНЫЕ ВОПРОСЫ: {smart_meta['smart_questions']}]"
        if smart_meta.get("smart_keywords"):
            final_text += f"\n[КЛЮЧЕВЫЕ СЛОВА: {smart_meta['smart_keywords']}]"

        meta = {
            'source_file': file_name, 'source_url': f"{base_url}{file_name}" if base_url else "",
            'page_title': page_title, 'release_version': release_version,    
            'equipment_type': equipment_type, 'library_name': library_key,
            'function_block_name': fb_name, 'breadcrumb_raw': breadcrumb_str,
            'format': 'html', 'source_type': source_type
        }
        
        if smart_meta.get("smart_keywords"):
            meta["keywords"] = smart_meta["smart_keywords"]
        if smart_meta.get("smart_questions"):
            meta["generated_questions"] = smart_meta["smart_questions"]

        meta = {k: v for k, v in meta.items() if v}
        return [Document(page_content=final_text, metadata=meta)]

    except Exception as e:
        logger.error(f"❌ Ошибка парсинга HTML {file_name}: {e}") # <--- ИСПОЛЬЗУЕМ ЛОГГЕР
        return []
