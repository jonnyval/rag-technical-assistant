"""
analyze_new_doc.py
==================
Скрипт для анализа нового DOCX-документа и поиска нестыковок
с существующими руководствами в векторной БД.

Использует:
1. Вашу готовую архитектуру UI (HTML-дашборд) и CLI.
2. Мощь RAGEngine (HyDE, Reranker, ротацию ключей) для точного поиска ответов.
3. Умные вопросы (smart_questions) из метаданных чанка для встречного допроса.

Запуск (из корня проекта):
    python scripts/analyze_new_doc.py --docx "путь/к/документу.docx"
"""

import sys
import os
import argparse
import html as _html
import datetime
from pathlib import Path
from typing import List, Dict

# ── LangChain & Pydantic ─────────────────────────────────────────────────────
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logger import log
from src.engine import RAGEngine, FUNCTION_CALLING_MODELS
from src.document_processing.parsers_qdrant_with_llm import process_docx_file


# =============================================================================
# 1. СХЕМА СТРУКТУРИРОВАННОГО ОТВЕТА
# =============================================================================

class DiscrepancyReport(BaseModel):
    """Описывает результат проверки нового фрагмента документации на расхождения с текущей базой знаний."""

    has_conflict: bool = Field(description="Есть ли фактические противоречия между новым и старым текстом?")
    details: str = Field(description="Подробное описание нестыковки. Если всё совпадает или это просто дополнение - пиши 'Нестыковок нет'.")
    severity: str = Field(description="Критичность: 'Low', 'Medium', 'High', 'None'")


# =============================================================================
# 2. НАРЕЗКА DOCX
# =============================================================================

def chunk_docx(docx_path: str, source_type: str) -> List[Dict]:
    """Разбирает DOCX на чанки и приводит метаданные к формату, удобному для сравнения через RAG."""

    images_tmp = "data/temp_analyze_images"
    os.makedirs(images_tmp, exist_ok=True)
    path = Path(docx_path)

    log.info(f"📄 Нарезка DOCX: {path.name} (source_type={source_type})")
    docs = process_docx_file(path, source_type=source_type, images_out_dir=images_tmp)

    chunks = []
    for doc in docs:
        meta = doc.metadata
        chunks.append({
            "title":               meta.get("page_title", "—"),
            "breadcrumb":          meta.get("breadcrumb_raw", ""),
            "equipment":           meta.get("equipment_type", ""),
            "generated_questions": meta.get("generated_questions", ""),
            "text":                doc.page_content,
        })

    log.info(f"✂️  Получено чанков: {len(chunks)}")
    return chunks


# =============================================================================
# 3. ИНИЦИАЛИЗАЦИЯ LLM ДЛЯ ОЦЕНЩИКА
# =============================================================================

def build_evaluator_chain():
    """Создает LLM-цепочку, которая оценивает расхождения между новым чанком и ответом из базы знаний."""

    provider = settings.active_llm
    log.info(f"🤖 Инициализация LLM Оценщика (evaluator): {provider} / {settings.llm_model_name}")

    # === Инициализация LLM в зависимости от провайдера (С РОТАЦИЕЙ КЛЮЧЕЙ) ===
    if provider == "gemini":
        if not settings.google_api_keys:
            raise ValueError("❌ GOOGLE_API_KEYS не установлена в .env!")
        
        llms = [
            ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                google_api_key=k,
                temperature=0.0,
                timeout=settings.llm_timeout,
            )
            for k in settings.google_api_keys
        ]
        
        if len(llms) > 1:
            llm = llms[0].with_fallbacks(llms[1:])
            log.info(f"✅ Ротация Gemini для evaluator включена ({len(llms)} ключей)")
        else:
            log.warning("⚠️  Только 1 Gemini ключ для evaluator — ротация отключена")
            llm = llms[0]
        
        method = "json_mode"
        
    elif provider == "gigachat":
        if not settings.gigachat_credentials:
            raise ValueError("❌ GIGACHAT_CREDENTIALS не установлена в .env!")
        llm = GigaChat(
            credentials=settings.gigachat_credentials,
            verify_ssl_certs=False,
            model=settings.llm_model_name,
            temperature=0.0
        )
        method = "json_mode"
        
    elif provider == "ollama":
        llm = ChatOpenAI(
            base_url=settings.ollama_url,
            api_key="ollama",
            model=settings.llm_model_name,
            temperature=0.0,
            timeout=settings.llm_timeout,
        )
        method = "json_mode"
        
    else:  # groq (по умолчанию)
        if not settings.groq_api_keys:
            raise ValueError("❌ GROQ_API_KEYS не установлена в .env!")
        
        llms = [
            ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=k,
                model=settings.llm_model_name,
                temperature=0.0,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )
            for k in settings.groq_api_keys
        ]
        
        if len(llms) > 1:
            llm = llms[0].with_fallbacks(llms[1:])
            log.info(f"✅ Ротация GROQ для evaluator включена ({len(llms)} ключей)")
        else:
            log.warning("⚠️  Только 1 GROQ ключ для evaluator — ротация отключена")
            llm = llms[0]
        
        method = "function_calling" if settings.llm_model_name in FUNCTION_CALLING_MODELS else "json_mode"

    prompt = ChatPromptTemplate.from_template("""
    Ты — эксперт по технической документации промышленных контроллеров.
    Твоя задача — сравнить выдержку из НОВОЙ версии документации со СТАРОЙ информацией из векторной БД.

    НОВЫЙ ЧАНК («{doc_title}», Раздел: {breadcrumb}):
    ---
    {new_chunk}
    ---

    ОТВЕТ ТЕКУЩЕЙ БАЗЫ ЗНАНИЙ (Старая версия):
    ---
    {old_rag_answer}
    ---

    Оцени, есть ли НЕСТЫКОВКИ, ПРОТИВОРЕЧИЯ или СУЩЕСТВЕННЫЕ РАСХОЖДЕНИЯ (в характеристиках, процедурах, терминологии).
    Внимание: если старая база отвечает, что информации нет, считай это дополнением документации, а НЕ противоречием (has_conflict = false).
    """)

    # === Structured Output с правильным методом ===
    log.debug(f"✓ Evaluator LLM инициализирована (метод: {method})")
    structured_llm = llm.with_structured_output(DiscrepancyReport, method=method)
    
    return prompt | structured_llm


# =============================================================================
# 4. АНАЛИЗ ОДНОЙ СЕКЦИИ ЧЕРЕЗ RAG ENGINE
# =============================================================================

def analyze_chunk(chunk: Dict, engine: RAGEngine, eval_chain, doc_title: str) -> Dict:
    """Сравнивает один чанк нового документа с текущей базой знаний и возвращает результат проверки."""

    # 1. Достаем сгенерированные парсером вопросы
    questions_str = chunk.get("generated_questions", "")
    if questions_str:
        questions = [q.strip() for q in questions_str.replace(';', ',').split(',') if len(q.strip()) > 10]
    else:
        questions = [chunk["breadcrumb"]] # Фоллбек

    all_conflicts = []
    all_sources = []
    has_any_conflict = False
    has_data = False
    
    # Берем до 2 самых важных вопросов, чтобы ускорить работу
    for q in questions[:2]:
        try:
            # 2. Спрашиваем текущую базу (RAGEngine применит HyDE и Reranker)
            eq_filter = [chunk.get("equipment")] if chunk.get("equipment") else None
            rag_response = engine.process_query(q, equipment_filter=eq_filter)
            
            if not rag_response.extracted_facts:
                continue
            
            has_data = True
            for fact in rag_response.extracted_facts:
                all_sources.append(fact.source_file)
                
            # 3. Сравниваем (rag_response.final_answer может быть пуст)
            final_answer = rag_response.final_answer if rag_response.final_answer else "[Нет данных]"
            
            evaluation = eval_chain.invoke({
                "doc_title": doc_title,
                "breadcrumb": chunk["breadcrumb"],
                "new_chunk": chunk["text"][:1500],
                "old_rag_answer": final_answer
            })
            
            if evaluation.has_conflict:
                has_any_conflict = True
                all_conflicts.append(f"**Вопрос: {q}**\n{evaluation.details}")
                
        except Exception as e:
            log.error(f"Ошибка LLM при проверке '{q}': {e}")
            all_conflicts.append(f"❌ Ошибка LLM: {str(e)[:200]}")
            
    # 4. Форматируем результат для HTML
    if all_conflicts:
        conflicts_text = "\n\n".join(all_conflicts)
    elif not has_data:
        conflicts_text = "⚠️ Похожих фрагментов в базе не найдено — нет возможности сравнить."
    else:
        conflicts_text = "НЕСТЫКОВОК НЕТ"
        
    return {
        "title":           chunk["title"],
        "breadcrumb":      chunk["breadcrumb"],
        "conflicts":       conflicts_text,
        "sources":         list(dict.fromkeys(all_sources)),
        "_has_conflict":   has_any_conflict,
        "_has_data":       has_data,
    }


# =============================================================================
# 5. ФОРМИРОВАНИЕ HTML-ОТЧЁТА (Ваша UI логика)
# =============================================================================

def _classify(result: Dict) -> tuple:
    """Назначает результату проверки тип, подпись и CSS-класс для HTML-отчета."""

    """Классифицирует результат анализа чанка по статусу.
    
    Returns:
        (severity, label, css_class): кортеж для сортировки и стилизации
        severity: 0=conflict, 1=nodata, 2=ok, 3=error
    """
    # Сначала проверяем структурированные флаги
    if "ошибка llm" in result["conflicts"].lower():
        return 3, "Ошибка", "error"
    
    if not result.get("_has_data", False):
        return 1, "Нет данных", "nodata"
    
    if result.get("_has_conflict", False):
        return 0, "Нестыковка", "conflict"
    
    return 2, "Нестыковок нет", "ok"

def build_report(results: List[Dict], docx_path: str, out_path: str):
    """Формирует интерактивный HTML-отчет по найденным конфликтам и пропускам в базе знаний."""

    doc_name = _html.escape(Path(docx_path).name)
    generated = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    classified = []
    for r in results:
        severity, label, css = _classify(r)
        classified.append({**r, "_severity": severity, "_label": label, "_css": css})
    classified.sort(key=lambda x: x["_severity"])

    counts = {
        "conflict": sum(1 for r in classified if r["_css"] == "conflict"),
        "nodata":   sum(1 for r in classified if r["_css"] == "nodata"),
        "ok":       sum(1 for r in classified if r["_css"] == "ok"),
        "error":    sum(1 for r in classified if r["_css"] == "error"),
    }

    cards_html = []
    for i, r in enumerate(classified):
        sources_html = ""
        if r.get("sources"):
            items = "".join(f'<li>{_html.escape(s)}</li>' for s in r["sources"])
            sources_html = f'<div class="sources"><span class="src-label">Источники из БД:</span><ul>{items}</ul></div>'

        conflict_text = _html.escape(r["conflicts"]).replace("\n", "<br>")
        breadcrumb_html = f'<div class="breadcrumb">{_html.escape(r["breadcrumb"])}</div>' if r.get("breadcrumb") else ""

        badge_icons = {"conflict": "⚡", "nodata": "◌", "ok": "✓", "error": "✕"}
        icon = badge_icons.get(r["_css"], "")

        cards_html.append(f'''
        <div class="card {r["_css"]}" data-severity="{r["_severity"]}" data-idx="{i}">
          <div class="card-header" onclick="toggleCard({i})">
            <span class="badge badge-{r["_css"]}">{icon} {r["_label"]}</span>
            <span class="card-title">{_html.escape(r["title"])}</span>
            <span class="chevron" id="chev-{i}">▾</span>
          </div>
          <div class="card-body" id="body-{i}">
            {breadcrumb_html}
            <div class="conflict-text">{conflict_text}</div>
            {sources_html}
          </div>
        </div>''')

    cards_joined = "\n".join(cards_html)

    # Весь CSS и JS из вашего исходного варианта
    html_out = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Анализ нестыковок — {doc_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0f1117; --surface: #181c27; --border: #262c3d; --text: #c9d1e0; --text-dim: #5a6480; --accent: #4f8ef7;
    --conflict-bg: #1e1215; --conflict-bd: #7c1f2e; --conflict-text: #f4637a;
    --nodata-bg: #141820; --nodata-bd: #2e3a5c; --nodata-text: #7b9bd4;
    --ok-bg: #0e1a14; --ok-bd: #1e4a2e; --ok-text: #4cba7a;
    --error-bg: #1a1510; --error-bd: #5c3a10; --error-text: #d4884c;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; line-height: 1.6; min-height: 100vh; }}
  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 28px 40px 20px; position: sticky; top: 0; z-index: 100; backdrop-filter: blur(8px); }}
  .header-top {{ display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
  .logo {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); letter-spacing: .12em; text-transform: uppercase; }}
  h1 {{ font-size: 18px; font-weight: 600; color: #e8edf5; }}
  .meta {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }}
  .counters {{ display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
  .cnt {{ display: flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; border: 1px solid transparent; transition: opacity .15s; font-family: 'IBM Plex Mono', monospace; }}
  .cnt:hover {{ opacity: .75; }}
  .cnt.conflict {{ background: var(--conflict-bg); border-color: var(--conflict-bd); color: var(--conflict-text); }}
  .cnt.nodata   {{ background: var(--nodata-bg);   border-color: var(--nodata-bd);   color: var(--nodata-text); }}
  .cnt.ok       {{ background: var(--ok-bg);       border-color: var(--ok-bd);       color: var(--ok-text); }}
  .cnt.error    {{ background: var(--error-bg);    border-color: var(--error-bd);    color: var(--error-text); }}
  .cnt.active   {{ box-shadow: 0 0 0 2px currentColor; }}
  .toolbar {{ padding: 12px 40px; background: var(--bg); border-bottom: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }}
  .search {{ flex: 1; max-width: 360px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-family: 'IBM Plex Mono', monospace; font-size: 13px; padding: 7px 12px; outline: none; transition: border-color .15s; }}
  .search:focus {{ border-color: var(--accent); }}
  .search::placeholder {{ color: var(--text-dim); }}
  .btn-expand {{ background: var(--surface); border: 1px solid var(--border); color: var(--text-dim); border-radius: 6px; padding: 7px 14px; font-size: 12px; cursor: pointer; font-family: 'IBM Plex Mono', monospace; transition: border-color .15s, color .15s; }}
  .btn-expand:hover {{ border-color: var(--accent); color: var(--accent); }}
  main {{ max-width: 960px; margin: 0 auto; padding: 28px 40px 60px; display: flex; flex-direction: column; gap: 8px; }}
  .card {{ border-radius: 8px; border: 1px solid var(--border); background: var(--surface); overflow: hidden; transition: border-color .15s; }}
  .card.conflict {{ border-color: var(--conflict-bd); background: var(--conflict-bg); }}
  .card.nodata   {{ border-color: var(--nodata-bd);   background: var(--nodata-bg); }}
  .card.ok       {{ border-color: var(--ok-bd);       background: var(--ok-bg); }}
  .card.error    {{ border-color: var(--error-bd);    background: var(--error-bg); }}
  .card.hidden   {{ display: none; }}
  .card-header {{ display: flex; align-items: center; gap: 10px; padding: 11px 16px; cursor: pointer; user-select: none; }}
  .card-header:hover {{ background: rgba(255,255,255,.03); }}
  .badge {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; white-space: nowrap; flex-shrink: 0; letter-spacing: .04em; }}
  .badge-conflict {{ background: rgba(124,31,46,.4); color: var(--conflict-text); }}
  .badge-nodata   {{ background: rgba(46,58,92,.5);  color: var(--nodata-text); }}
  .badge-ok       {{ background: rgba(30,74,46,.4);  color: var(--ok-text); }}
  .badge-error    {{ background: rgba(92,58,16,.4);  color: var(--error-text); }}
  .card-title {{ flex: 1; font-size: 13px; font-weight: 500; color: #d0d8ea; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .chevron {{ color: var(--text-dim); font-size: 14px; transition: transform .2s; flex-shrink: 0; }}
  .chevron.open {{ transform: rotate(180deg); }}
  .card-body {{ padding: 0 16px 16px; display: none; border-top: 1px solid rgba(255,255,255,.04); }}
  .card-body.open {{ display: block; }}
  .breadcrumb {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-dim); padding: 10px 0 8px; border-bottom: 1px solid rgba(255,255,255,.04); margin-bottom: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .conflict-text {{ font-size: 13px; line-height: 1.7; color: var(--text); white-space: pre-wrap; word-break: break-word; }}
  .sources {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.05); }}
  .src-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-dim); font-family: 'IBM Plex Mono', monospace; }}
  .sources ul {{ list-style: none; margin-top: 6px; display: flex; flex-direction: column; gap: 3px; }}
  .sources li {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); padding: 3px 8px; background: rgba(79,142,247,.07); border-radius: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  #no-results {{ display: none; text-align: center; color: var(--text-dim); padding: 40px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
<header>
  <div class="header-top">
    <span class="logo">RegLab · Анализ документации</span>
    <h1>{doc_name}</h1>
  </div>
  <div class="meta">Сгенерировано: {generated} &nbsp;·&nbsp; Всего чанков: {len(results)}</div>
  <div class="counters">
    <div class="cnt conflict active" onclick="filterBy('conflict')">⚡ Нестыковки <strong>{counts["conflict"]}</strong></div>
    <div class="cnt nodata" onclick="filterBy('nodata')">◌ Нет данных <strong>{counts["nodata"]}</strong></div>
    <div class="cnt ok" onclick="filterBy('ok')">✓ Чисто <strong>{counts["ok"]}</strong></div>
    <div class="cnt error" onclick="filterBy('error')">✕ Ошибки <strong>{counts["error"]}</strong></div>
  </div>
</header>
<div class="toolbar">
  <input class="search" id="search" type="text" placeholder="Поиск по заголовку или тексту…" oninput="filterSearch()">
  <button class="btn-expand" onclick="expandAll()">Развернуть все</button>
  <button class="btn-expand" onclick="collapseAll()">Свернуть все</button>
</div>
<main id="cards">
{cards_joined}
  <div id="no-results">Ничего не найдено</div>
</main>
<script>
  let activeFilter = 'conflict';
  const cards = Array.from(document.querySelectorAll('.card[data-severity]'));
  function toggleCard(idx) {{
    const body = document.getElementById('body-' + idx);
    const chev = document.getElementById('chev-' + idx);
    const open = body.classList.toggle('open');
    chev.classList.toggle('open', open);
  }}
  function expandAll() {{
    cards.filter(c => !c.classList.contains('hidden')).forEach(c => {{
      const idx = c.dataset.idx;
      document.getElementById('body-' + idx).classList.add('open');
      document.getElementById('chev-' + idx).classList.add('open');
    }});
  }}
  function collapseAll() {{
    cards.forEach(c => {{
      const idx = c.dataset.idx;
      document.getElementById('body-' + idx).classList.remove('open');
      document.getElementById('chev-' + idx).classList.remove('open');
    }});
  }}
  function filterBy(cls) {{
    if (activeFilter === cls) {{
      activeFilter = null;
      document.querySelectorAll('.cnt').forEach(c => c.classList.remove('active'));
    }} else {{
      activeFilter = cls;
      document.querySelectorAll('.cnt').forEach(c => {{
        c.classList.toggle('active', c.classList.contains(cls));
      }});
    }}
    applyFilters();
  }}
  function filterSearch() {{ applyFilters(); }}
  function applyFilters() {{
    const q = document.getElementById('search').value.toLowerCase();
    let visible = 0;
    cards.forEach(card => {{
      const matchFilter = !activeFilter || card.classList.contains(activeFilter);
      const text = card.textContent.toLowerCase();
      const matchSearch = !q || text.includes(q);
      const show = matchFilter && matchSearch;
      card.classList.toggle('hidden', !show);
      if (show) visible++;
    }});
    document.getElementById('no-results').style.display = visible ? 'none' : 'block';
  }}
  document.addEventListener('DOMContentLoaded', () => {{
    applyFilters();
    cards.filter(c => c._css !== 'hidden' && c.classList.contains('conflict'))
         .slice(0, 3)
         .forEach(c => {{
           const idx = c.dataset.idx;
           document.getElementById('body-' + idx).classList.add('open');
           document.getElementById('chev-' + idx).classList.add('open');
         }});
  }});
</script>
</body>
</html>"""

    Path(out_path).write_text(html_out, encoding="utf-8")
    print(f"\n✅ Готово! Отчёт: {out_path}")
    print(f"   ⚡ Нестыковки: {counts['conflict']} | ✓ Чисто: {counts['ok']} | ◌ Нет данных: {counts['nodata']}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    """Точка входа CLI: разбирает новый DOCX, проверяет его через RAG и сохраняет HTML-отчет."""

    parser = argparse.ArgumentParser(description="Анализ нового DOCX против векторной БД")
    parser.add_argument("--docx", required=True, help="Путь к анализируемому DOCX")
    parser.add_argument(
        "--source_type", default="r500s_docx",
        choices=["r500s_docx", "r500_docx", "astraregul_docx", "metod_docx"],
        help="Тип документа — определяет style_map для mammoth",
    )
    parser.add_argument("--out", default="report_conflicts.html", help="Путь для HTML-отчёта")
    parser.add_argument("--equipment", default=None, help="Фильтр по типу оборудования в БД (напр. R500S)")
    parser.add_argument("--chunks", default=None, type=int, help="Анализировать только первые N чанков")
    args = parser.parse_args()

    if not Path(args.docx).exists():
        print(f"❌ Файл не найден: {args.docx}")
        sys.exit(1)

    doc_title = Path(args.docx).stem

    # Шаг 1: Нарезка
    print(f"\n✂️  Шаг 1/3: Нарезка {args.docx} (source_type={args.source_type})...")
    chunks = chunk_docx(args.docx, source_type=args.source_type)
    
    if args.chunks:
        chunks = chunks[: args.chunks]
        print(f"   Ограничено до {args.chunks} чанков (--chunks)")

    if not chunks:
        print("❌ Парсер не вернул чанков.")
        sys.exit(1)

    # Шаг 2: Инициализация ядра (Используем RAGEngine)
    print("\n🔌 Шаг 2/3: Подключение ядра RAG (Qdrant + Reranker + HyDE)...")
    try:
        engine = RAGEngine()
        eval_chain = build_evaluator_chain()
    except Exception as e:
        print(f"❌ Ошибка инициализации ядра: {e}")
        sys.exit(1)

    # Шаг 3: Анализ
    print(f"\n🤖 Шаг 3/3: Встречный допрос по {len(chunks)} чанкам...")
    results = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk["breadcrumb"] or chunk["title"]
        print(f"   [{i}/{len(chunks)}] {label[:70]}...")
        result = analyze_chunk(chunk, engine, eval_chain, doc_title)
        results.append(result)

    # Шаг 4: Отчёт
    build_report(results, args.docx, args.out)


if __name__ == "__main__":
    main()
