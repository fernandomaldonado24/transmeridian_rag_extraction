# =========================
# Standard libraries
# =========================
import os
import re
import base64
import tempfile
import time
from datetime import datetime
from pathlib import Path
import logging

# =========================
# Third-party libraries
# =========================
import fitz
import camelot
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from unstract.llmwhisperer import LLMWhispererClientV2
# =========================
# Env + Clients
# =========================
load_dotenv()

openai_client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    api_key=os.getenv("OPENAI_API_KEY")
)

LLM_WHISPERER_API_KEY = os.getenv("LLM_WHISPERER_PRO_API_KEY_EU")
LLM_WHISPERER_BASE_URL = os.getenv("BASE_URL_EU")
gpt_vision_model = "gpt-4.1"
# =========================
# Helpers
# =========================

# =========================
# Token counting
# =========================
ENCODING = tiktoken.get_encoding("cl100k_base")
TABLE_TOKEN_THRESHOLD = 4000

# Count tokens using OpenAI-compatible tokenizer (cl100k_base)
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

# =========================
# Table chunking
# =========================

# Split a large table into row-based chunks while preserving headers
# Ensures each chunk stays below the token threshold
def split_table_by_rows(headers: str, rows: list[str], max_tokens: int):
    chunks = []
    current_rows = []

    for row in rows:
        candidate = headers + "\n" + "\n".join(current_rows + [row])

        if count_tokens(candidate) > max_tokens and current_rows:
            chunks.append(headers + "\n" + "\n".join(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(headers + "\n" + "\n".join(current_rows))

    return chunks

# =========================
# ASCII → Markdown tables
# =========================
# Convert ASCII-style tables into Markdown
# Automatically splits large tables into token-safe chunks
# Each chunk preserves headers to maintain semantic integrity
def ascii_table_to_md(block: str) -> str:
    rows = []

    for line in block.splitlines():
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cols)

    if not rows:
        return block

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    if len(rows) >= 2 and rows[0][0] == "" and rows[1][0] == "":
        merged = []
        for c1, c2 in zip(rows[0], rows[1]):
            merged.append((c1 + " " + c2).strip())
        rows = [merged] + rows[2:]

    md = []
    md.append("| " + " | ".join(rows[0]) + " |")
    md.append("| " + " | ".join(["---"] * width) + " |")

    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")

    table_text = "\n".join(md)

    if count_tokens(table_text) <= TABLE_TOKEN_THRESHOLD:
        return table_text

    headers = md[0] + "\n" + md[1]
    body_rows = md[2:]

    return "\n\n".join(
        split_table_by_rows(headers, body_rows, TABLE_TOKEN_THRESHOLD)
    )

# Reconstruct full page text by detecting ASCII tables and converting them to Markdown
def rebuild_text_with_tables(extraction):
    text = extraction["result_text"].splitlines()

    out = []
    buffer = []
    inside_table = False

    for line in text:
        if "+" in line and "-" in line:
            inside_table = True
            buffer.append(line)
            continue

        if inside_table:
            if line.strip() == "":
                # end of table
                out.append(ascii_table_to_md("\n".join(buffer)))
                buffer = []
                inside_table = False
                out.append(line)
            else:
                buffer.append(line)
        else:
            out.append(line)

    # catch last table
    if buffer:
        out.append(ascii_table_to_md("\n".join(buffer)))

    return "\n".join(out)
# =========================
# Extractors
# =========================

# Send a single-page PDF to LLM Whisperer, reconstruct tables, and return clean text
def call_llm_whisperer(page_bytes: bytes):
    # write temp 1-page pdf
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(page_bytes)
        tmp_path = tmp.name
    # LLM Whisperer client
    client = LLMWhispererClientV2(
        base_url=LLM_WHISPERER_BASE_URL,
        api_key=LLM_WHISPERER_API_KEY
    )

    # Silence WHISPERER LOGS
    for name in [
        "unstract",
        "unstract.llmwhisperer",
        "unstract.llmwhisperer.client_v2",
    ]:
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).disabled = True

    for attempt in range(3):
        try:
            result = client.whisper(
                file_path=tmp_path,
                wait_for_completion=True,
                mode="table",
                wait_timeout=300  # increase
            )
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(10)

    os.remove(tmp_path)

    if result.get("status") != "processed":
        raise RuntimeError(result.get("message"))

    extraction = result["extraction"]

    # Reconstruct tables before any chunking happens
    clean_text = rebuild_text_with_tables(extraction)

    return clean_text

# Extract plain text from a single-page PDF using PyMuPDF
def reader(page_bytes: bytes):
    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]
    text = page.get_text("text")
    doc.close()
    return text.strip()

# ================ GPT VISION CALL ================
# Initialize OpenAI client
openai_client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    api_key=os.getenv("OPENAI_API_KEY")
)
gpt_vision_model = "gpt-4.1"

# NEW - for full document reading
def reader_full_doc(pdf_path: str):
    """Extract plain text from entire PDF"""
    doc = fitz.open(pdf_path)
    page_texts = []
    for page in doc:
        page_texts.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(page_texts)

# Convert a PDF page to image and use GPT Vision to extract tables as Markdown
def gpt_vision(page_bytes: bytes):
    # convert PDF page to image
    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    doc.close()

    b64 = base64.b64encode(img_bytes).decode()

    prompt = """
            Extract ALL tables from this page.
            Return ONLY markdown tables.
            Preserve rows, columns, headers exactly.
            Do not add commentary.
        """

    response = openai_client.responses.create(
        model=gpt_vision_model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_base64": b64,
                    "mime_type": "image/png"
                }
            ]
        }]
    )

    return response.output_text.strip()
# =========================
# Page Classifier
# =========================

# Classify a PDF page to decide which extraction method to use based on layout signals
def classify_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # --- PyMuPDF signals ---
    text_len = len(page.get_text("text"))
    image_count = len(page.get_images())
    drawing_count = len(page.get_drawings())

    def label(name):
        metrics = f"text_len:{text_len} | image_count:{image_count} | drawing_count:{drawing_count}"
        return name, metrics

    # --- Rule 1: SCANNED OR TABLE WITH NO GRIDS ---
    if text_len < 40 and image_count >= 1:
        return label("WHISPERER")
    if drawing_count > 64 and text_len > 800:  # There is likely a table with no grids or an important chart
        return label("WHISPERER")

    # --- Rule 2: TABLE (Camelot detector) ---
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_number + 1),  # Camelot is 1-indexed
            flavor="lattice"
        )
        if len(tables) == 0:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number + 1),
                flavor="stream"
            )
        if len(tables) > 0:
            return label("VISION")
    except Exception:
        pass

    # --- Default ---
    return label("PDF_READER")
# =========================
# PDF Processor
# =========================
# NEW FUNCTION
def process_pdfs(pdf_folder, output_folder, companies=None):
    output_folder.mkdir(exist_ok=True)

    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_companies = 0
    total_pages = 0

    candidates = []
    for p in pdf_folder.glob("*.pdf"):
        if companies and p.stem not in companies:
            continue
        if (output_folder / f"{p.stem}.txt").exists():
            print(f"Skipping {p.stem} (already processed)")
            continue
        candidates.append(p)

    total_companies_target = len(candidates)
    total_pages_target = sum(fitz.open(p).page_count for p in candidates)

    print(f"Starting processing at {start_dt}")
    print(f"Companies to process: {len(candidates)}")

    for pdf_file in candidates:
        total_companies += 1
        print(f"Processing: {pdf_file.name}")

        doc = fitz.open(pdf_file)
        page_count = doc.page_count
        page_results = [None] * page_count
        total_pages += page_count

        blocks = [
            list(range(i, min(i + 10, page_count)))
            for i in range(0, page_count, 10)
        ]

        for block in blocks:
            multi = fitz.open()
            for page_index in block:
                multi.insert_pdf(doc, from_page=page_index, to_page=page_index)

            pdf_bytes = multi.tobytes()
            multi.close()

            block_text = call_llm_whisperer(pdf_bytes)
            split_pages = block_text.split("\f")

            for idx, page_index in enumerate(block):
                page_text = split_pages[idx] if idx < len(split_pages) else ""
                page_results[page_index] = f"\n\n--- PAGE {page_index+1} ---\n\n{page_text}"
                
        doc.close()

        output_path = output_folder / f"{pdf_file.stem}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(page_results))

        print(f"Saved: {pdf_file.name}")
        print(
            f"Progress: "
            f"Companies {total_companies / total_companies_target * 100:.1f}% | "
            f"Pages {total_pages / total_pages_target * 100:.1f}%"
        )

    end_time = time.time()
    duration_min = (end_time - start_time) / 60

    print("\n===== RUN SUMMARY =====")
    print(f"Start time: {start_dt}")
    print(f"Total companies: {total_companies}")
    print(f"Total pages: {total_pages}")
    print(f"Total time (min): {duration_min:.2f}")


# OLD FUNCTION
'''
# Process all PDFs in a folder, classify each page, apply the correct extractor, and save results
def process_pdfs(pdf_folder, output_folder, companies=None):
    output_folder.mkdir(exist_ok=True)

    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # === COUNTERS ===
    total_companies = 0
    total_pages = 0

# === CANDIDATES (PDFs eligible for processing: filtered by company list and skipping already processed outputs) ===
    candidates = []

    for p in pdf_folder.glob("*.pdf"):
        if companies and p.stem not in companies:
            continue

        if (output_folder / f"{p.stem}.txt").exists():
            print(f"Skipping {p.stem} (already processed)")
            continue

        candidates.append(p)

    # === DENOMINATORS ===
    total_companies_target = len(candidates)
    total_pages_target = 0

    for pdf_file in candidates:
        with fitz.open(pdf_file) as doc:
            total_pages_target += doc.page_count
    
    print(f"Starting processing at {start_dt}")
    print(f"Companies to process: {len(candidates)}")
    # === MAIN LOOP ===
    for pdf_file in candidates:
        total_companies += 1
        print(f"Processing: {pdf_file.name}")

        doc = fitz.open(pdf_file)
        page_texts = []

        for i, _ in enumerate(doc):
            total_pages += 1

            method, metrics = classify_page(pdf_file, i)

            single = fitz.open()
            single.insert_pdf(doc, from_page=i, to_page=i)
            page_bytes = single.tobytes()
            single.close()

            if method == "PDF_READER":
                text = reader(page_bytes)
            elif method == "WHISPERER":
                text = call_llm_whisperer(page_bytes)
            elif method == "VISION":
                text = gpt_vision(page_bytes)
            else:
                text = ""

            page_texts.append(
                f"\n\n--- PAGE {i+1} | {method} | {metrics} ---\n\n{text}"
            )

        doc.close()

        output_path = output_folder / f"{pdf_file.stem}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(page_texts))

        print(f"Saved: {pdf_file.name}")


        # === PROGRESS AFTER THIS COMPANY ===
        companies_progress = (total_companies / total_companies_target) * 100
        pages_progress = (total_pages / total_pages_target) * 100

        print(
            f"Progress: "
            f"Companies {total_companies / total_companies_target * 100:.1f}% | "
            f"Pages {total_pages / total_pages_target * 100:.1f}%"
        )

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_min = (end_time - start_time) / 60

    print("\n===== RUN SUMMARY =====")
    print(f"Start time: {start_dt}")
    print(f"End time:   {end_dt}")
    print(f"Companies processed: {total_companies}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total time (minutes): {duration_min:.2f}")
'''