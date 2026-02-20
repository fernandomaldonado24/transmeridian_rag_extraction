import os
import time
from datetime import datetime, timedelta
import pandas as pd
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import psycopg2  # Assuming you are using PostgreSQL
import paramiko
from sshtunnel import SSHTunnelForwarder
import json
import logging
import boto3
from botocore.exceptions import ClientError
import uuid
from openai import OpenAI
from google import genai
from google.genai import types
import tiktoken
from pypdf import PdfReader
import numpy as np
import tiktoken
from pypdf import PdfReader
from bs4 import BeautifulSoup
from pathlib import Path

ENCODING = tiktoken.get_encoding("cl100k_base")

instructions = (
"""
IMPORTANT

Return the most recent reporting year available, include the year in DETAILS.

If no emissions were found, return:

VALUE: <0>
DETAILS: <brief explanation>

RULES:
- For every DETAIL look carefully for the page where the information has been taken, the page number is at the end of the page.
- Page number can be in the format "- i -" (e.g. "- 1 -", "- 10 -") or "Page i" (e.g. "Page 1", "Page 7").
- Use only emissions expressed in tCO2e. If the unit is different, convert the figure to tCO2e (tons).
- For Scope 2, prioritise Market-based values.
- If scopes are combined (e.g., Scope 1 & 2) and cannot be separated, return VALUE: 0.
- If emissions are broken into categories, sum them. If a total is provided, use the total.
- Ignore intensity ratios, percentages, targets, and narrative text.

OUTPUT FORMAT (STRICT):

VALUE: <number>
DETAILS: <brief explanation and page number>

"""
        )

# '''
# OBJECTIVE

# Answer 10 questions and always return a numeric value in VALUE and a brief description in DETAILS.

# RULES

# Return the most recent reporting year available.

# For emissions, return the most recent reported emissions value and reference the page where the information was found.

# The page number must follow the specific format: "- i -" (e.g., "- 1 -", "- 2 -").

# If no information relevant to the question is found, return:

# VALUE: <0>

# DETAILS: <No data found>

# Always convert emissions to tCO2 and use that value in VALUE.

# If emissions are broken down into categories, sum them.

# If a total emissions value is provided, use the total.

# If Total Emissions is not explicitly provided, calculate it as:
# Scope 1 + Scope 2 (location-based) + Scope 3.

# OUTPUT FORMAT (STRICT)
# VALUE: <number>
# DETAILS: <brief explanation and page number>
# '''


openai_model = "gpt-5-mini"
gemini_model = "gemini-2.0-flash-001"

# --- Module logger ------------------------------------------------------------
logger = logging.getLogger("esg")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# Function to list and filter files
def get_files(path):
    files = [f for f in os.listdir(path) if f != ".DS_Store"]
    print(f"Found {len(files)} files in {path}")
    return files

# Function to upload files to OpenAI and get file IDs
def upload_files_openai(client, path, files):
    file_ids = []
    for i, file in enumerate(files, start=1):
        file_path = os.path.join(path, file)
        print(f"Uploading {i}/{len(files)}: {file} to OpenAI...")
        try:
            with open(file_path, "rb") as f:
                file_created = client.files.create(
                    file=f,
                    purpose="user_data"
                )
        except Exception as e:
            print(f"Γ¥î Failed to upload {file}: {e}")
            continue
        file_ids.append(file_created.id)
    print("All files uploaded to OpenAI.")
    return file_ids
# # Function to create OpenAI vector store and assistant
# def create_openai_assistant(client, file_ids, company_name, instructions):
#     print("Creating OpenAI vector store...")
#     vector_store = client.vector_stores.create(
#         name=f"Triage - {company_name} - Metrics 3 - V6",
#         file_ids=file_ids
#     )
#     print("Creating OpenAI assistant...")
#     my_assistant = client.beta.assistants.create(
#         instructions=instructions,
#         name=f"Triage Analyst - {company_name} - Metrics 3 - V6",
#         tools=[{"type": "file_search"}],
#         tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
#         model="gpt-4o-mini"
#     )
#     print("OpenAI assistant created successfully.")
#     return my_assistant.id

# Function to process questions using OpenAI
def process_questions_openai(client, instructions, file_ids, questions, model=openai_model, retries=3, retry_delay=3, max_workers=5):
    results = {}
    # Handle DataFrame input
    if isinstance(questions, pd.DataFrame):
        if questions.empty:
            print("No questions provided, skipping OpenAI processing.")
            return results
        # Convert DataFrame rows to named tuples for easier attribute access
        # itertuples() creates named tuples with column names as attributes
        questions = list(questions.itertuples(index=False))
    else:
        # Check if questions is empty (for list/other iterables)
        try:    
            if len(questions) == 0:
                print("No questions provided, skipping OpenAI processing.")
                return results
        except (TypeError, AttributeError):
            if not questions:
                print("No questions provided, skipping OpenAI processing.")
                return results
    total_questions = len(questions)
    print(f"Processing {total_questions} questions with Responses API...")
    def process_single_question(index, question):
        for attempt in range(retries):
            try:
                print(f"({index}/{total_questions}) Processing: {question.label} (Attempt {attempt + 1})")
                content = [{"type": "input_text", "text": question.content}] + [
                    {"type": "input_file", "file_id": f} for f in file_ids
                ]
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=[{"role": "user", "content": content}]
                )
                return resp
            except Exception as e:
                print(f"Error processing {question.label}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_question = {
            executor.submit(process_single_question, idx, question): question
            for idx, question in enumerate(questions, start=1)
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            resp = future.result()
            if resp is not None:
                results[question.label] = resp
    print("Responses API question processing complete.")
    return results

# Function to extract OpenAI responses
def extract_openai_responses(results, questions):
    import re
    import pandas as pd
    print("Extracting OpenAI responses...")
    response_rows = []
    # Handle DataFrame input
    if isinstance(questions, pd.DataFrame):
        # Convert DataFrame rows to named tuples for easier attribute access
        questions = list(questions.itertuples(index=False))
    def _extract_output_text(resp):
        # Prefer the simple output_text list provided by the Responses API
        output_text_candidates = getattr(resp, "output_text", None)
        if isinstance(output_text_candidates, list):
            for candidate in output_text_candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        # Fall back to scanning the structured output payload
        outputs = getattr(resp, "output", None) or []
        for entry in outputs:
            if isinstance(entry, dict):
                content_items = entry.get("content")
            else:
                content_items = getattr(entry, "content", None)
            if not content_items:
                continue
            if not isinstance(content_items, list):
                content_items = [content_items]
            for item in content_items:
                if isinstance(item, dict):
                    content_type = item.get("type")
                    text_value = item.get("text") or item.get("value")
                else:
                    content_type = getattr(item, "type", None)
                    text_value = getattr(item, "text", None) or getattr(item, "value", None)
                if content_type in {"output_text", "text"} and text_value:
                    return text_value.strip()
        # Last resort: check plain text attribute
        resp_text = getattr(resp, "text", None)
        if isinstance(resp_text, str):
            return resp_text.strip()
        return ""
    for index, question in enumerate(questions, start=1):
        resp = results.get(question.label)
        if not resp or not getattr(resp, "output", None):
            print(f"No response for question: {question.label}")
            continue
        try:
            output_text = _extract_output_text(resp)
        except Exception as e:
            print(f"Error extracting text for question {question.label}: {e}")
            output_text = ""
        score = 0
        try:
            if output_text:
                last_word = output_text.split(" ")[-1]
                match = re.match(r"\d+(\.\d+)?$", last_word)
                if match:
                    score = float(match.group())
        except Exception as e:
            print(f"Error parsing score for question {question.label}: {e}")
        response_rows.append({
            "question_number": index,
            "question_text": question.content,
            "openai_score": score,
            "openai_response_text": output_text
        })
    print("OpenAI responses extracted.")
    return pd.DataFrame(response_rows)

# GEMINI FUNCTIONS
# TOKEN PADDING FUNCTIONS
MIN_CACHE_TOKENS = 4096
SAFETY_BUFFER = 200  # extra tokens to avoid off-by-one errors
def generate_padding_text(target_tokens: int, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    base_sentence = (
        "This section is intentionally added as neutral padding text "
        "to meet minimum context size requirements for cached content. "
    )
    tokens = []
    while len(tokens) < target_tokens:
        tokens.extend(encoding.encode(base_sentence))
    # Trim exactly to target token count
    return encoding.decode(tokens[:target_tokens])

def extract_pdfs_to_txt_files(pdf_folder_path: str, output_folder_path: str = None):
    """
    Extract text from PDFs in a folder and save each PDF's text as a .txt file.
    Useful for preparing files for the chunking/embedding approach.
    
    :param pdf_folder_path: Path to folder containing PDF files
    :param output_folder_path: Path to folder where .txt files will be saved. 
                               If None, saves .txt files in the same folder as PDFs.
    :return: List of paths to created .txt files
    """
    if output_folder_path is None:
        output_folder_path = pdf_folder_path
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    txt_files_created = []
    skipped_image_pdfs = []
    
    for file_name in os.listdir(pdf_folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue
        
        pdf_path = os.path.join(pdf_folder_path, file_name)
        # Create output .txt filename (same name, different extension)
        txt_filename = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_folder_path, txt_filename)
        
        try:
            reader = PdfReader(pdf_path)
            all_text_parts = []
            file_text_found = False
            
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    file_text_found = True
                    all_text_parts.append(text)
            
            if not file_text_found:
                skipped_image_pdfs.append(file_name)
                print(f"⚠️ Skipped image-based PDF: {file_name}")
            else:
                # Combine all pages into one text file
                full_text = "\n\n".join(all_text_parts)
                # Save to .txt file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                txt_files_created.append(txt_path)
                print(f"✅ Extracted text from {file_name} → {txt_filename}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
    
    print(f"\n📄 Created {len(txt_files_created)} .txt files in {output_folder_path}")
    if skipped_image_pdfs:
        print(f"⚠️ Skipped {len(skipped_image_pdfs)} image-based PDF(s)")
    
    return txt_files_created

def extract_text_and_pad(folder_path: str, encoding_name: str = "cl100k_base"):
    """
    Extract text from PDFs in folder, skip image-based PDFs, pad to minimum token count.
    Returns list of text parts suitable for Gemini `Content`.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    all_text_parts = []
    total_tokens = 0
    skipped_image_pdfs = []
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(folder_path, file_name)
        reader = PdfReader(file_path)
        file_text_found = False
        file_tokens = 0
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                file_text_found = True
                all_text_parts.append(text)
                tokens = len(encoding.encode(text))
                total_tokens += tokens
                file_tokens += tokens
        if not file_text_found:
            skipped_image_pdfs.append(file_name)
            print(f"⚠️ Skipped image-based PDF: {file_name}")
        else:
            print(f"{file_name}: {file_tokens} tokens")
    print(f"\nReal content tokens: {total_tokens}")
    # Pad if below Gemini minimum
    if total_tokens < MIN_CACHE_TOKENS:
        missing = MIN_CACHE_TOKENS - total_tokens + SAFETY_BUFFER
        print(f"Padding with ~{missing} tokens")
        padding_text = generate_padding_text(missing, encoding_name)
        all_text_parts.append(
            "\n\n[BEGIN NEUTRAL PADDING — IGNORE FOR ANALYSIS]\n"
            + padding_text +
            "\n[END NEUTRAL PADDING]\n"
        )
        total_tokens += missing
    print(f"Final token count (approx): {total_tokens}")
    if skipped_image_pdfs:
        print("\n📄 Image-based PDFs skipped:")
        for pdf in skipped_image_pdfs:
            print(f" - {pdf}")
    return all_text_parts

def upload_and_cache_gemini(
    client,
    path: str,
    files: list[str],
    instructions: str,
    ttl_minutes: int = 120,
    ):
    """
    Upload PDF files to Gemini and create a cached content object.
    Skips image-based PDFs, counts tokens, pads if too small.
    """
    if not isinstance(instructions, str):
        raise ValueError("instructions must be a string")
    if not files:
        print("❌ No files provided — skipping cache creation.")
        return None
    print(f"📤 Uploading {len(files)} PDF files to Gemini...")
    # Step 1: Extract text + pad
    all_text_parts = extract_text_and_pad(path)
    if not all_text_parts:
        print("❌ No valid content to cache — skipping.")
        return None
    # Step 2: Convert to Gemini `Content` objects
    documents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=t)]  # <- pass text as keyword
        )
        for t in all_text_parts
    ]
    ttl_seconds = ttl_minutes * 60
    ttl_string = f"{ttl_seconds}s"
    print("🧠 Creating Gemini cached content...")
    cache = client.caches.create(
        model=gemini_model,
        config=types.CreateCachedContentConfig(
            system_instruction=instructions,
            contents=documents,
            ttl=ttl_string,
        ),
    )
    print(f"✅ Cached content created: {cache.name}")
    return cache.name

def process_questions_gemini(
    client,
    cache_name: str | None,
    questions: list,
):
    """
    Process a list of questions using Gemini.
    Uses cached content if cache_name is provided.
    """
    # Handle DataFrame input (avoids "truth value of a DataFrame is ambiguous")
    if isinstance(questions, pd.DataFrame):
        if questions.empty:
            return pd.DataFrame()
        questions = list(questions.itertuples(index=False))
    else:
        # Handle list/iterables safely without relying on truthiness
        try:
            if len(questions) == 0:
                return pd.DataFrame()
        except (TypeError, AttributeError):
            if questions is None:
                return pd.DataFrame()
            # If it's a non-sized iterable, materialize it once
            questions = list(questions)
            if len(questions) == 0:
                return pd.DataFrame()

    mode = "with cached content" if cache_name else "without cached content"
    print(f"🤖 Processing {len(questions)} questions {mode}...")
    def process_question(index, question):
        config_kwargs = {}
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        response = client.models.generate_content(
            model=gemini_model,
            contents=[question.content],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None) or getattr(response, "content", "")
        if not text:
            return None
        # Extract numeric score from the final token (your existing convention)
        match = re.search(r"\d+(\.\d+)?$", text.strip())
        score = float(match.group()) if match else 0.0
        return {
            "question_number": index,
            "question_text": question.content,
            "gemini_score": score,
            "gemini_response_text": text,
        }
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda x: process_question(*x),
            enumerate(questions, start=1),
        )
    print("✅ Gemini question processing complete.")
    return pd.DataFrame([r for r in results if r is not None])

# NEW VERSION FOR EMISSIONS
def save_to_excel(openai_df, company_name):
    print("Saving emissions results to Excel...")

    def parse(text):
        if pd.isna(text):
            return "", ""

        value_line = re.search(r"VALUE:\s*(.+)", text)
        details_line = re.search(r"DETAILS:\s*(.+)", text)

        value_raw = value_line.group(1) if value_line else ""

        # Extract only the number, ignore units
        number_match = re.search(r"[\d,\.]+", value_raw)
        value = number_match.group(0) if number_match else ""

        details = details_line.group(1).strip() if details_line else ""

        return value, details

    openai_df[["openai_value", "openai_details"]] = \
        openai_df["openai_response_text"].apply(lambda x: pd.Series(parse(x)))

    result = openai_df[[
        "question_text",
        "openai_value",
        "openai_details"
    ]]

    result.insert(0, "company_number", company_name)

    file_name = "Emissions Results (Market based).xlsx"

    if Path(file_name).exists():
        existing = pd.read_excel(file_name)
        result = pd.concat([existing, result], ignore_index=True)

    result.to_excel(file_name, index=False)

    print(f"Excel saved/updated: {file_name}")

# OLD M3 VERSION
'''
def save_to_excel(openai_df, gemini_df, company_name, gemini_highlight_list, openai_highlight_list, conn, company_assessment_id):
    print("Merging responses and saving to Excel...")
    combined_responses = pd.merge(openai_df, gemini_df, on=['question_number', 'question_text'], how='outer')
    combined_responses['openai_score'] = combined_responses['openai_score'].astype(float)
    combined_responses['gemini_score'] = combined_responses['gemini_score'].astype(float)
    # Add empty 'manual_score' column
    combined_responses.insert(2, 'manual_score', '') 
    # Add 'AI_score' column based on engine of choice
    def determine_ai_score(row):
        question_number = row['question_number']
        if question_number in gemini_highlight_list:
            return row['gemini_score']
        elif question_number in openai_highlight_list:
            return row['openai_score']
        else:
            return ''  # or you could return max or average of the two scores
    # Add AI_score column after manual_score
    combined_responses.insert(3, 'AI_score', combined_responses.apply(determine_ai_score, axis=1))
    # Reorder columns with the new AI_score column
    column_order = [
        "question_number", "question_text", "manual_score", "AI_score",
        "openai_score", "gemini_score", "openai_response_text", "gemini_response_text"
    ]
    combined_responses = combined_responses[column_order]
    # Define file name
    file_name = f"Triage - {company_name} - Metrics 3.xlsx"
    # Handle local mode (no database connection)
    is_local_mode = (conn is None or company_assessment_id is None)
    if is_local_mode:
        # Local mode: skip database query and use simplified structure
        excel_df = combined_responses.rename(columns = {"question_number": "OldQuestionNumber"})
        result = excel_df.copy()
        # Add placeholder columns for consistency (in the same order as database mode)
        result.insert(0, 'AssessmentQuestionID', None)
        result.insert(0, 'CompanyAssessmentID', None)
        # Ensure column order matches database mode
        result = result[["CompanyAssessmentID", "AssessmentQuestionID", "OldQuestionNumber", "question_text", "manual_score", "AI_score", "openai_score", "gemini_score", "openai_response_text", "gemini_response_text"]]
    else:
        # Database mode: get scores from database
        portal_scores = rds_get_metrics_3_scores(conn, company_assessment_id)
        excel_df = combined_responses.rename(columns = {"question_number": "OldQuestionNumber"})
        # merge the two dataframes
        result = pd.merge(excel_df, portal_scores, on='OldQuestionNumber', how='left')
        # select only relevant columns
        result = result[["AssessmentQuestionID","OldQuestionNumber", "question_text", "manual_score", "AI_score", "openai_score", "gemini_score", "openai_response_text", "gemini_response_text"]]
        # Add company_assessment_id column
        result.insert(0, 'CompanyAssessmentID', company_assessment_id) 
    # Save to Excel with formatting
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        result.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        # Define formats
        green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})  # Green for Gemini
        blue_format = workbook.add_format({'bg_color': '#CCE5FF', 'font_color': '#003366'})   # Blue for OpenAI
        chosen_format = workbook.add_format({'bold': True})  # Format for the AI_score column
        # Get column indexes
        gemini_score_col_index = result.columns.get_loc("gemini_score")
        openai_score_col_index = result.columns.get_loc("openai_score")
        ai_score_col_index = result.columns.get_loc("AI_score")
        # Apply formatting for each question number
        for row_num, question_number in enumerate(result["OldQuestionNumber"], start=1):  # start=1 to match Excel row numbers
            # Highlight engine scores
            if question_number in gemini_highlight_list:
                worksheet.write(row_num, gemini_score_col_index, result.at[row_num - 1, "gemini_score"], green_format)
                # Also make AI_score bold if it's from Gemini
                worksheet.write(row_num, ai_score_col_index, result.at[row_num - 1, "AI_score"], chosen_format)
            if question_number in openai_highlight_list:
                worksheet.write(row_num, openai_score_col_index, result.at[row_num - 1, "openai_score"], blue_format)
                # Also make AI_score bold if it's from OpenAI
                worksheet.write(row_num, ai_score_col_index, result.at[row_num - 1, "AI_score"], chosen_format)
    print(f"Excel file '{file_name}' saved successfully.")
    return(result)
'''

# Function to insert automated assessment responses into the database    
def insert_assessment_responses(cursor, questions, openai_df, gemini_df):
    for index, row in openai_df.iterrows():
        question = questions[index]  # Get the corresponding question
        prompt_id = question.PromptID
        engine_id = 1  # OpenAI responses have EngineID = 1
        openai_score = row['openai_score']
        openai_response_text = row['openai_response_text']
        # 'Used' is True only if the engine_id matches the question's engine_id
        used = question.EngineID == engine_id  
        # Insert OpenAI response
        insert_query = """
            INSERT INTO "AssessmentQuestionResponse" 
            ("EngineID", "ResponseText", "Used", "Score", "PromptID")
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (engine_id, openai_response_text, used, openai_score, prompt_id))
    for index, row in gemini_df.iterrows():
        question = questions[index]  # Get the corresponding question
        prompt_id = question.PromptID
        engine_id = 2  # Gemini responses have EngineID = 2
        gemini_score = row['gemini_score']
        gemini_response_text = row['gemini_response_text']
        # 'Used' is True only if the engine_id matches the question's engine_id
        used = question.EngineID == engine_id  
        # Insert Gemini response
        insert_query = """
            INSERT INTO "AssessmentQuestionResponse" 
            ("EngineID", "ResponseText", "Used", "Score", "PromptID")
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (engine_id, gemini_response_text, used, gemini_score, prompt_id))  

# Function to save responses to the database
def save_to_database(conn, questions, openai_df, gemini_df):
    try:
        with conn.cursor() as cursor:
            insert_assessment_responses(cursor, questions, openai_df, gemini_df)
            conn.commit()  # Commit the transaction
            print("Responses inserted successfully into AssessmentQuestionResponse.")
    except Exception as e:
        print(f"Error inserting into database: {e}")
        conn.rollback()  # Rollback in case of error

# Function to get prompts from the database
def rds_get_prompts(conn, survey_id=3333):
    Question = namedtuple(
        "Question",
        ["label", "content", "run_id", "thread_id", "PromptID", "EngineID", "QuestionID"]
    )
    with conn.cursor() as cursor:
        if survey_id == 1111:
            # Metrics L1+
            query = """
                SELECT q."QuestionFull", NULL AS PromptID, 1 AS EngineID, q."QuestionID"
                FROM "SurveyQuestion" sq
                LEFT JOIN "Question" q ON sq."QuestionID" = q."QuestionID"
                WHERE sq."SurveyID" = %s
            """
            cursor.execute(query, (survey_id,))
        elif survey_id == 3333:
            # Metrics L2/L3
            query = """
                SELECT p."PromptQuestion", p."PromptID", p."EngineID", q."QuestionID"
                FROM "Prompt" p
                LEFT JOIN "Question" q ON p."PromptID" = q."PromptID"
                ORDER BY q."OldQuestionNumber"
            """
            cursor.execute(query)
        else:
            return []
        rows = cursor.fetchall()
    # Γ£à Single unified construction of `questions`
    questions = [
        Question(
            label=f"Question {i+1}",
            content=row[0],
            run_id=None,
            thread_id=None,
            PromptID=int(row[1]) if row[1] is not None else None,
            EngineID=int(row[2]) if row[2] is not None else None,
            QuestionID=int(row[3]) if row[3] is not None else None
        )
        for i, row in enumerate(rows)
    ]
    return questions

# Function to get number of employees ID
def rds_get_numberofemployeesid(conn, raw_employees):
    # Build the SQL query to find the matching employee count bucket using BETWEEN
    query = f'''
        SELECT "NumberOfEmployeeID"
        FROM "NumberOfEmployee"
        WHERE {raw_employees} BETWEEN "RangeStart" AND "RangeEnd"
        LIMIT 1
    '''
    # Execute the query
    with conn.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    # Check if a result was found
    if result is None:
        print("Warning: No matching employee count bucket found for the given employee count.")
        return None
    # Return the first result
    return result[0]

# Function to get revenue ID
def rds_get_revenueid(conn, raw_revenue):
    # Build the SQL query to find the matching revenue bucket using BETWEEN
    query = f'''
        SELECT "CompanyRevenueID"
        FROM "CompanyRevenue"
        WHERE {raw_revenue} BETWEEN "StartRevenue" AND "EndRevenue"
        LIMIT 1
    '''
    # Execute the query
    with conn .cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    # Check if a result was found
    if result is None:
        print("Warning: No matching revenue bucket found for the given turnover.")
        return None
    # Return the first result
    return result[0]

# Function to get company ID
def rds_get_or_create_company_id(conn, company_name, sector_id, industry_id, numberofemployees_id, revenue_id, company_website=None):
    """
    Fetch the CompanyID for a given company name. If no match is found, create a new company record.
    :param conn: A psycopg2 database connection object.
    :param company_name: The name of the company to search for or insert.
    :param company_website: (Optional) The website of the company.
    :return: The CompanyLeadID (existing or newly created).
    """
    # Generate a company secret 
    secret = str(uuid.uuid4())  # Generates a random UUID (version 4)
    print(secret)
    if not company_name:
        logging.error("Invalid company name provided.")
        return None
    try:
        with conn.cursor() as cur:
            # Try to find an existing company ID
            query = """
                SELECT "CompanyLeadID" FROM public."CompanyLead"
                WHERE LOWER("CompanyName") = LOWER(%s)
                LIMIT 1;
            """
            cur.execute(query, (company_name,))
            result = cur.fetchone()
            if result:
                logging.info("Successfully retrieved the company ID.")
                return result[0]
            # If no match, insert a new record (handle optional website)
            if company_website:
                insert_query = """
                    INSERT INTO public."Company" ("CompanyName", "CompanyWebSite", "Secret", "SectorID", "IndustryID", "NumberOfEmployeeID", "CompanyRevenueID")
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING "CompanyID";
                """
                cur.execute(insert_query, (company_name, company_website, secret, sector_id, industry_id, numberofemployees_id, revenue_id))
            else:
                insert_query = """
                    INSERT INTO public."Company" ("CompanyName")
                    VALUES (%s) RETURNING "CompanyID";
                """
                cur.execute(insert_query, (company_name,))
            new_company_id = cur.fetchone()[0]
            conn.commit()
            logging.info("New company record created.")
            return new_company_id
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return None

# Function to get product ID    
def rds_get_product_id(conn, product_description):
    if not product_description:
        logging.error("Invalid product_description provided.")
        return None
    try:
        with conn.cursor() as cur:
            # Try to find an existing ProductID
            query = """
                SELECT "ProductID" FROM public."Product"
                WHERE "ProductDescription" = %s
                LIMIT 1;
            """
            cur.execute(query, (product_description,))
            result = cur.fetchone()
            if result:
                logging.info("Successfully retrieved the existing ProductID.")
                return result[0]
            # Explicitly return None if no match is found
            return None
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return None

# Function to get survey ID
def rds_get_survey_id(conn, product_id):
    if not product_id:
        logging.error("Invalid product_description provided.")
        return None
    try:
        with conn.cursor() as cur:
            # Try to find an existing ProductID
            query = """
                SELECT "SurveyID" FROM public."Survey"
                WHERE "ProductID" = %s
                LIMIT 1;
            """
            cur.execute(query, (product_id,))
            result = cur.fetchone()
            if result:
                logging.info("Successfully retrieved the existing SurveyID.")
                return result[0]
            # Explicitly return None if no match is found
            return None
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return None

# Function to get status ID    
def rds_get_status_id(conn, status_description):
    if not status_description:
        logging.error("Invalid product_description provided.")
        return None
    try:
        with conn.cursor() as cur:
            # Try to find an existing ProductID
            query = """
                SELECT "AssessmentStatusID" FROM public."AssessmentStatus"
                WHERE "AssessmentStatusDescription" = %s
                LIMIT 1;
            """
            cur.execute(query, (status_description,))
            result = cur.fetchone()
            if result:
                logging.info("Successfully retrieved the existing AssessmentStatusID.")
                return result[0]
            # Explicitly return None if no match is found
            return None
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return None

# Function to update company assessment status
def rds_update_companyassessment_status(conn, company_assessment_id, status_id):
    """
    Update the AssessmentStatusID for a given CompanyAssessmentID.
    :param conn: A psycopg2 database connection object.
    :param company_assessment_id: The ID of the company assessment to update.
    :param status_id: The new status ID to set.
    :return: True if the update was successful, False otherwise.
    """
    if not company_assessment_id or not status_id:
        logging.error("Invalid company_assessment_id or status_id provided.")
        return False
    try:
        with conn.cursor() as cur:
            # Update the assessment status
            query = """
                UPDATE public."CompanyAssessment"
                SET "AssessmentStatusID" = %s
                WHERE "CompanyAssessmentID" = %s;
            """
            cur.execute(query, (status_id, company_assessment_id))
            if cur.rowcount > 0:
                conn.commit()
                logging.info("Successfully updated the CompanyAssessment status.")
                return True
            else:
                logging.warning("No matching CompanyAssessmentID found. No update performed.")
                return False
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return False

# Function to update company assessment status
def rds_update_companyassessment_status(conn, company_assessment_id, status_id):
    """
    Update the AssessmentStatusID for a given CompanyAssessmentID.
    :param conn: A psycopg2 database connection object.
    :param company_assessment_id: The ID of the company assessment to update.
    :param status_id: The new status ID to set.
    :return: True if the update was successful, False otherwise.
    """
    if not company_assessment_id or not status_id:
        logging.error("Invalid company_assessment_id or status_id provided.")
        return False
    try:
        with conn.cursor() as cur:
            # Update the assessment status
            query = """
                UPDATE public."CompanyAssessment"
                SET "AssessmentStatusID" = %s
                WHERE "CompanyAssessmentID" = %s;
            """
            cur.execute(query, (status_id, company_assessment_id))
            if cur.rowcount > 0:
                conn.commit()
                logging.info("Successfully updated the CompanyAssessment status.")
                return True
            else:
                logging.warning("No matching CompanyAssessmentID found. No update performed.")
                return False
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return False

def rds_insert_into_assessmentquestion(conn, questions, company_assessment_id):
    assessment_question_ids = {}  # Dictionary to store the AssessmentQuestionID for each prompt or question
    with conn.cursor() as cursor:
        for question in questions:
            # Use PromptID if it exists, otherwise fall back to QuestionID
            prompt_id = getattr(question, 'PromptID', None) or question.QuestionID
            question_id = question.QuestionID
            print(f"Inserting Question: {question_id}, Prompt: {prompt_id}")  # Debugging print
            insert_query = """
                INSERT INTO "AssessmentQuestion" 
                ("QuestionID", "CompanyAssessmentID")
                VALUES (%s, %s) RETURNING "AssessmentQuestionID"
            """
            try:
                cursor.execute(insert_query, (question_id, company_assessment_id))
                result = cursor.fetchone()
                if result:
                    assessment_question_ids[prompt_id] = result[0]  # Store by PromptID if available, else QuestionID
                else:
                    print(f"No AssessmentQuestionID returned for prompt {prompt_id}")
            except Exception as e:
                print(f"Error inserting prompt {prompt_id}: {e}")
            conn.commit()
    return assessment_question_ids

def rds_insert_assessment_responses(conn, questions, openai_df=None, gemini_df=None, assessment_question_ids=None, engines=('openai', 'gemini')):
    with conn.cursor() as cursor:
        # Insert OpenAI responses if selected
        if 'openai' in engines and openai_df is not None:
            for index, row in openai_df.iterrows():
                question = questions[index]
                # Key for lookup in assessment_question_ids: PromptID if present, else QuestionID
                lookup_key = getattr(question, 'PromptID', None) or question.QuestionID
                engine_id = 1
                openai_score = row['openai_score']
                openai_response_text = row['openai_response_text']
                assessment_question_id = assessment_question_ids[lookup_key]
                # For L1+ (survey 1111), PromptID is None and there is no row in Prompt;
                # we MUST NOT stuff QuestionID into the PromptID column or we break FK constraints.
                db_prompt_id = getattr(question, 'PromptID', None)
                used = getattr(question, 'EngineID', None) == engine_id
                insert_query = """
                    INSERT INTO "AssessmentQuestionResponse" 
                    ("EngineID", "ResponseText", "Used", "Score", "PromptID", "AssessmentQuestionID")
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (engine_id, openai_response_text, used, openai_score, db_prompt_id, assessment_question_id))
        # Insert Gemini responses if selected
        if 'gemini' in engines and gemini_df is not None:
            for index, row in gemini_df.iterrows():
                question = questions[index]
                # Key for lookup in assessment_question_ids: PromptID if present, else QuestionID
                lookup_key = getattr(question, 'PromptID', None) or question.QuestionID
                engine_id = 2
                gemini_score = row['gemini_score']
                gemini_response_text = row['gemini_response_text']
                assessment_question_id = assessment_question_ids[lookup_key]
                db_prompt_id = getattr(question, 'PromptID', None)
                used = getattr(question, 'EngineID', None) == engine_id
                insert_query = """
                    INSERT INTO "AssessmentQuestionResponse" 
                    ("EngineID", "ResponseText", "Used", "Score", "PromptID", "AssessmentQuestionID")
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (engine_id, gemini_response_text, used, gemini_score, db_prompt_id, assessment_question_id))
    conn.commit()

def rds_get_used_questionassessmentresponses(conn, company_assessment_id, engine_id='both'):
    # Handle 'both' string - treat it as None to get all used responses
    if engine_id == 'both':
        engine_id = None
    if engine_id == 1:
        print("Fetching all OpenAI responses for the given CompanyAssessmentID.")
    elif engine_id == 2:
        print("Fetching all Gemini responses for the given CompanyAssessmentID.")
    else:
        print("Fetching only used responses across all engines for the given CompanyAssessmentID.")
    # Base query
    query = """
        SELECT aqr."AssessmentQuestionID", aqr."ResponseText", aqr."Score"
        FROM "AssessmentQuestionResponse" aqr
        LEFT JOIN "AssessmentQuestion" aq ON aqr."AssessmentQuestionID" = aq."AssessmentQuestionID"
        WHERE aq."CompanyAssessmentID" = %s
    """
    params = [company_assessment_id]
    if engine_id is not None:
        query += " AND aqr.\"EngineID\" = %s"
        params.append(engine_id)
    else:
        query += " AND aqr.\"Used\" = TRUE"
    query += " ORDER BY aqr.\"PromptID\" ASC"
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        results = cursor.fetchall()
    responses = [
        {
            "AssessmentQuestionID": row[0],
            "ResponseText": row[1],
            "Score": int(row[2])
        }
        for row in results
    ]
    return responses

def rds_check_for_existing_companyassessment_record(conn, company_id, product_id):
    with conn.cursor() as cur:
        query = f"SELECT * FROM \"CompanyAssessment\" WHERE \"CompanyID\" = {company_id} AND \"ProductID\" = {product_id}"
        cur.execute(query)
        companyassessment_record = cur.fetchall()
        if len(companyassessment_record) == 0:
            print("No existing CompanyAssessment record found.")
            return False 
        else:
            print("Existing CompanyAssessment record found.")
            companyassessment_id = companyassessment_record[0][0]
            return companyassessment_id

def rds_update_assessmentquestions_with_used_responses(conn, company_assessment_id, engine_id='both'):
    # Fetch responses from AssessmentQuestionResponse table
    responses = rds_get_used_questionassessmentresponses(conn, company_assessment_id, engine_id)
    if not responses:
        print("No responses found for the given CompanyAssessmentID.")
        return
    # Update each AssessmentQuestion with AnswerText and Score
    update_query = """
        UPDATE "AssessmentQuestion"
        SET "AnswerText" = %s, "Score" = %s
        WHERE "AssessmentQuestionID" = %s
    """
    try:
        with conn.cursor() as cursor:
            for response in responses:
                cursor.execute(update_query, (response["ResponseText"], response["Score"], response["AssessmentQuestionID"]))
            conn.commit()
        print("Successfully updated AssessmentQuestion table.")
    except Exception as e:
        conn.rollback()
        print(f"Error updating AssessmentQuestion table: {e}")

def rds_get_metrics_3_scores(conn, company_assessment_id):
    # Handle local mode (no database connection)
    if conn is None or company_assessment_id is None:
        print("Local mode: Skipping database query for Metrics 3 scores")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["AssessmentQuestionID", "OldQuestionNumber"])
    # Fetch scores from AssessmentQuestion table
    query = """ SELECT aq."AssessmentQuestionID", q."OldQuestionNumber",p."PromptQuestion", aq."AnswerText", aq."Score", aq."CompanyAssessmentID" FROM "AssessmentQuestion" aq
    LEFT JOIN "Question" q ON aq."QuestionID" = q."QuestionID"
    LEFT JOIN "Prompt" p ON q."PromptID" = p."PromptID"
    WHERE "CompanyAssessmentID" = %(company_assessment_id)s
    ORDER BY q."QuestionID" ASC
    """     
    results = pd.DataFrame(columns=["AssessmentQuestionID", "OldQuestionNumber"])  # Initialize with default
    try:
        results = pd.read_sql_query(query, conn, params={"company_assessment_id": company_assessment_id})
        print("Successfully got Metrics 3 scores")
    except Exception as e:
        print(f"Error getting Metrics 3 scores: {e}")
        # Return empty DataFrame if query fails
        results = pd.DataFrame(columns=["AssessmentQuestionID", "OldQuestionNumber"])
    return results

def rds_get_company_secret(conn, company_id):
    query = """
        SELECT "Secret" FROM "Company" WHERE "CompanyID" = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (company_id,))
        result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return "No secret found"

def rds_get_sector_id(conn, sector_desc):
    query = """
        SELECT "SectorID" FROM "Sector" WHERE "SectorDescription" = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (sector_desc,))
        result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return "No sector found"

def rds_correct_ai_scores_in_assessmentquestion(conn, company_scores, companyassessment_id, excel = True): 
    changes_counter = 0
    changes_df = pd.DataFrame(columns=['CompanyAssessmentID', 'AssessmentQuestionID', 'OldScore', 'NewScore'])
    for index, rows in company_scores.iterrows():
        question_number = rows['question_number']
        print(f"the question number is {question_number}")
        manual_score = rows['manual_score']
        print(manual_score)
        ai_score = rows['Score']
        print(ai_score)
        if manual_score == ai_score:
            print("The manual score and AI score are the same. No action needed.")
        else:
            print("The manual score and AI score are different. Updating the AssessmentQuestion table to reflect correct scores.")
            assessmentquestion_id = rows['AssessmentQuestionID']
            print(f"CompanyAssessmentID: {companyassessment_id}")
            print(f"AssessmentQuestionID: {assessmentquestion_id}")
            changes_counter += 1 
            # Add the new row to changes_df using pd.concat()
            new_row = pd.DataFrame([{
                'CompanyAssessmentID': companyassessment_id,
                'AssessmentQuestionID': assessmentquestion_id,
                'OldScore': ai_score,
                'NewScore': manual_score
            }])
            changes_df = pd.concat([changes_df, new_row], ignore_index=True)
            with conn.cursor() as cur:
                query = f"UPDATE \"AssessmentQuestion\" SET \"Score\" = {manual_score} WHERE \"CompanyAssessmentID\" = {companyassessment_id} AND \"AssessmentQuestionID\" = {assessmentquestion_id}"
                cur.execute(query, (manual_score, companyassessment_id, assessmentquestion_id))
                conn.commit()
    print(f"Total number of changes made: {changes_counter}")
    print(changes_df)
    if len(changes_df) == 0:
        print("No changes were made. No excel file will be created.")
        excel = False # don't create an excel if no changes were made.
    if excel == True:
        changes_df.to_excel(f"{companyassessment_id}_changes_made.xlsx", index=False)
    return(changes_df)

def rds_get_question_id_for_metrics3_based_on_oldquestionnumber(conn, old_question_number):
    with conn.cursor() as cur:
        query = f"SELECT * FROM \"Question\" q LEFT JOIN \"SurveyQuestion\" sq ON q.\"QuestionID\" = sq.\"QuestionID\"  WHERE sq.\"SurveyID\" = 3333 AND \"OldQuestionNumber\" = {old_question_number}"
        cur.execute(query)
        question_id = cur.fetchall()
        question_id = question_id[0][0]
    return question_id

def S3_upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket
    :param file_name: File to upload - make this a file_path to the company like so : file_path = f"{company_name}/{file_name}"
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used - remember to include the Metrics 3.0 subfolder - like so f"{company_secret} /Metrics 3.0 Documents/2025-03-24/{file_name}"
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)
     # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def rds_insert_manual_scores_into_assessmentquestion(conn, scores):
    for index, rows in scores.iterrows():
        companyassessment_id = rows["CompanyAssessmentID"]
        print(f"Company Assessment ID: {companyassessment_id}")
        old_question_number = rows['question_number']
        print(f"Old question number: {old_question_number}")
        question_id = rds_get_question_id_for_metrics3_based_on_oldquestionnumber(conn, old_question_number)
        manual_score = rows['manual_score']
        print(f"Manual score: {manual_score}")
        with conn.cursor() as cur:
            query = "INSERT INTO \"AssessmentQuestion\" (\"QuestionID\", \"Score\", \"CompanyAssessmentID\" ) VALUES (%s, %s, %s)"
            cur.execute(query, (question_id, manual_score, companyassessment_id))
            conn.commit()
    return ("Scores inserted successfully.")

# --- Small validators ---------------------------------------------------------
def _coerce_int(name, value):
    if value is None:
        raise ValueError(f"{name} must not be None")
    try:
        iv = int(value)
    except Exception:
        raise ValueError(f"{name} must be an integer-like value (got {value!r})")
    if iv <= 0:
        raise ValueError(f"{name} must be a positive integer (got {iv})")
    return iv

def _coerce_int_list(name, values):
    if values is None:
        raise ValueError(f"{name} must not be None")
    try:
        ints = [int(v) for v in values]
    except Exception:
        raise ValueError(f"All {name} must be integer-like (got {values!r})")
    ints = [v for v in ints if v > 0]
    if not ints:
        raise ValueError(f"{name} must contain at least one positive integer")
    # de-duplicate (preserve order)
    seen = set()
    dedup = []
    for v in ints:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup

def _ensure_conn(conn):
    if conn is None or not hasattr(conn, "cursor"):
        raise ValueError("conn must be a valid DB-API connection object")

def activate_company_assessment(conn, company_assessment_id, rescore=False):
    """
    Activates a pending company assessment (status 1 -> 2) and deactivates any existing 
    active assessments (status 2 -> 3 or 4) for the same company, product, and survey.
    Args:
        conn: Database connection
        company_assessment_id: The CompanyAssessmentID to activate
        rescore: If True, marks old assessments as Invalid (4); if False, marks as Inactive (3)
    Returns:
        dict with activation results or None if failed
    """
    _ensure_conn(conn)
    company_assessment_id = _coerce_int("company_assessment_id", company_assessment_id)
    sql = """
        SELECT * FROM activate_company_assessment(%s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (company_assessment_id, rescore))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.commit()
            if not row:
                logger.warning("No result from activate_company_assessment for CompanyAssessmentID=%s", company_assessment_id)
                return None
            result = dict(zip(cols, row))
            # Log the activation
            deactivated_count = len(result.get('deactivated_assessment_ids', [])) if result.get('deactivated_assessment_ids') else 0
            status_type = "Invalid (4)" if rescore else "Inactive (3)"
            if deactivated_count > 0:
                logger.info(
                    "Activated CompanyAssessmentID %s and marked %d previous assessment(s) as %s",
                    company_assessment_id, deactivated_count, status_type
                )
            else:
                logger.info(
                    "Activated CompanyAssessmentID %s (no previous active assessments found)",
                    company_assessment_id
                )
            return result
    except Exception as e:
        logger.exception("Failed to activate CompanyAssessmentID=%s", company_assessment_id)
        try: conn.rollback()
        except Exception: pass
        raise

def rds_update_survey_score(conn, companyassessment_id):
    """
    Updates public."CompanyAssessment" using values from view_assessmentcompanyscore:
      - "CompanyScore" <- view."CompanySectorScore" (stores sector score in CompanyScore field)
      - "UpdateDate"   <- CURRENT_DATE
    Returns a 1-row DataFrame: CompanyAssessmentID, CompanyScore, CompanyScoreSectorID.
    """
    _ensure_conn(conn)
    companyassessment_id = _coerce_int("companyassessment_id", companyassessment_id)
    sql = """
    WITH src AS (
      SELECT
        ca."CompanyAssessmentID",
        v."CompanySectorScore"::numeric(7,2) AS "CompanySectorScore"
      FROM public."CompanyAssessment" ca
      JOIN public.view_assessmentcompanyscore v
        ON v."CompanyAssessmentID" = ca."CompanyAssessmentID"
      WHERE ca."CompanyAssessmentID" = %s
    )
    UPDATE public."CompanyAssessment" ca
    SET "CompanyScore" = src."CompanySectorScore",
        "UpdateDate" = CURRENT_DATE
    FROM src
    WHERE ca."CompanyAssessmentID" = src."CompanyAssessmentID"
    RETURNING ca."CompanyAssessmentID", ca."CompanyScore", ca."CompanyScoreSectorID";
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (companyassessment_id,))
            row = cur.fetchone()
            conn.commit()
        if not row:
            logger.warning("No view row for CompanyAssessmentID=%s", companyassessment_id)
            return None
        # Round CompanyScore to 2 decimal places to match numeric(7,2) precision
        rounded_row = (row[0], round(row[1], 2) if row[1] is not None else None, row[2])
        return pd.DataFrame([rounded_row], columns=["CompanyAssessmentID", "CompanyScore", "CompanyScoreSectorID"])
    except Exception as e:
        logger.exception("Failed to update survey score for CompanyAssessmentID=%s", companyassessment_id)
        try: conn.rollback()
        except Exception: pass
        raise

def rds_update_survey_scores_many(conn, companyassessment_ids):
    """
    Batch update using view_assessmentcompanyscore for many assessments.
    Stores view.CompanySectorScore into CompanyAssessment.CompanyScore field.
    Returns DataFrame columns: CompanyAssessmentID, CompanyScore, CompanyScoreSectorID.
    """
    _ensure_conn(conn)
    ids = _coerce_int_list("companyassessment_ids", companyassessment_ids)
    if not ids:
        return pd.DataFrame(columns=["CompanyAssessmentID", "CompanyScore", "CompanyScoreSectorID"])
    sql = """
    WITH src AS (
      SELECT
        ca."CompanyAssessmentID",
        v."CompanySectorScore"::numeric(7,2) AS "CompanySectorScore"
      FROM public."CompanyAssessment" ca
      JOIN public.view_assessmentcompanyscore v
        ON v."CompanyAssessmentID" = ca."CompanyAssessmentID"
      WHERE ca."CompanyAssessmentID" = ANY(%s)
    )
    UPDATE public."CompanyAssessment" ca
    SET "CompanyScore" = src."CompanySectorScore",
        "UpdateDate" = CURRENT_DATE
    FROM src
    WHERE ca."CompanyAssessmentID" = src."CompanyAssessmentID"
    RETURNING ca."CompanyAssessmentID", ca."CompanyScore", ca."CompanyScoreSectorID";
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (ids,))
            rows = cur.fetchall()
            conn.commit()
        if not rows:
            logger.warning("No view rows returned for IDs=%s", ids)
            return pd.DataFrame(columns=["CompanyAssessmentID", "CompanyScore", "CompanyScoreSectorID"])
        # Round CompanyScore to 2 decimal places to match numeric(7,2) precision
        rounded_rows = [(row[0], round(row[1], 2) if row[1] is not None else None, row[2]) for row in rows]
        df = pd.DataFrame(rounded_rows, columns=["CompanyAssessmentID", "CompanyScore", "CompanyScoreSectorID"])
        # Optional: warn if some IDs werenΓÇÖt updated
        missing = sorted(set(ids) - set(df["CompanyAssessmentID"].astype(int).tolist()))
        if missing:
            logger.warning("Some CompanyAssessmentIDs had no view row: %s", missing)
        return df
    except Exception:
        logger.exception("Failed batch survey score update for IDs=%s", ids)
        try: conn.rollback()
        except Exception: pass
        raise

def rds_insert_or_update_category_scores(conn, companyassessment_id):
    """
    Upsert AssessmentCategory.SummaryScore for a given CompanyAssessmentID and
    stamp/refresh CompleteDate.
    Behavior:
    - Uses public.get_category_weight_scores(bigint) to compute per-category scores.
    - If an AssessmentCategory row already exists, updates SummaryScore and sets CompleteDate = NOW().
    - If it doesn't exist, inserts a new row with CompleteDate = NOW().
    - Returns a DataFrame of the updated/inserted rows joined to Category for descriptions.
    """
    # Step 1: Existing AssessmentCategory rows for this assessment
    select_query = """
        SELECT "AssessmentCategoryID", "CategoryID"
        FROM public."AssessmentCategory"
        WHERE "CompanyAssessmentID" = %s;
    """
    with conn.cursor() as cur:
        cur.execute(select_query, (companyassessment_id,))
        category_rows = cur.fetchall()
        # Map CategoryID -> AssessmentCategoryID
        category_mapping = {row[1]: row[0] for row in category_rows}
    # Step 2: Calculate category scores via updated SQL function (one param)
    score_query = """
        SELECT "CategoryID", "CategoryWeightScore"
        FROM public.get_category_weight_scores(%s::bigint);
    """
    with conn.cursor() as cur:
        cur.execute(score_query, (companyassessment_id,))
        category_scores = {row[0]: row[1] for row in cur.fetchall()}
    if not category_scores:
        print("No category scores calculated.")
        return None
    # Step 3A: Update existing entries (also refresh CompleteDate)
    update_query = """
        UPDATE public."AssessmentCategory"
        SET "SummaryScore" = %s,
            "CompleteDate" = NOW()
        WHERE "AssessmentCategoryID" = %s;
    """
    # Step 3B: Insert new entries if missing (with CompleteDate)
    insert_query = """
        INSERT INTO public."AssessmentCategory"
            ("CompanyAssessmentID", "CategoryID", "SummaryScore", "CompleteDate")
        VALUES (%s, %s, %s, NOW());
    """
    try:
        with conn.cursor() as cur:
            for category_id, summary_score in category_scores.items():
                if category_id in category_mapping:
                    # Update existing row
                    cur.execute(update_query, (summary_score, category_mapping[category_id]))
                else:
                    # Insert new row
                    cur.execute(insert_query, (companyassessment_id, category_id, summary_score))
        conn.commit()
        print("Category scores upserted successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error upserting category scores: {e}")
        raise
    # Step 4: Fetch updated results
    fetch_query = """
        SELECT
            ac."AssessmentCategoryID",
            cat."CategoryDescription",
            ac."SummaryScore",
            ac."CompanyAssessmentID",
            ac."CompleteDate"
        FROM public."AssessmentCategory" ac
        LEFT JOIN public."Category" cat
          ON ac."CategoryID" = cat."CategoryID"
        WHERE ac."CompanyAssessmentID" = %(companyassessment_id)s
        ORDER BY ac."AssessmentCategoryID";
    """
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"companyassessment_id": companyassessment_id})
        print("Successfully fetched updated category scores.")
    except Exception as e:
        print(f"Error fetching updated category scores: {e}")
        results = None
    return results

def rds_insert_or_update_topic_scores(conn, companyassessment_id):
    """
    Updates Topic Scores in the AssessmentTopic table if entries exist.
    If no entries exist, inserts new topic scores.
    """
    # Step 1: Get existing topics for the given AssessmentID
    select_query = """
        SELECT "AssessmentTopicID", "AssessmentCategoryID", "TopicID"
        FROM public."AssessmentTopic"
        WHERE "AssessmentCategoryID" IN (
            SELECT "AssessmentCategoryID"
            FROM public."AssessmentCategory"
            WHERE "CompanyAssessmentID" = %s
        );
    """
    with conn.cursor() as cur:
        cur.execute(select_query, (companyassessment_id,))
        topic_rows = cur.fetchall()  # Fetch existing topic IDs
        topic_mapping = {(row[1], row[2]): row[0] for row in topic_rows}  
        # {(AssessmentCategoryID, TopicID): AssessmentTopicID}
    # Step 2: Calculate topic scores using the SQL helper function
    # This delegates the aggregation logic to public.get_topic_weight_scores,
    # which returns CompanyAssessmentID, CategoryID, TopicID, TopicWeightScore.
    # We then join to AssessmentCategory to resolve AssessmentCategoryID.
    score_query = """
        SELECT
            ac."AssessmentCategoryID",
            t."TopicID",
            t."TopicWeightScore"
        FROM public.get_topic_weight_scores(ARRAY[%s]) AS t
        JOIN public."AssessmentCategory" ac
          ON ac."CompanyAssessmentID" = t."CompanyAssessmentID"
         AND ac."CategoryID" = t."CategoryID";
    """
    topic_scores = {}
    with conn.cursor() as cur:
        cur.execute(score_query, (companyassessment_id,))
        topic_scores = {(row[0], row[1]): row[2] for row in cur.fetchall()}  
        # {(AssessmentCategoryID, TopicID): Score}
    if not topic_scores:
        print("No topic scores calculated.")
        return None
    # Step 3A: Update existing topics
    update_query = """
        UPDATE public."AssessmentTopic"
        SET "Score" = %s
        WHERE "AssessmentTopicID" = %s;
    """
    # Step 3B: Insert new topics if missing
    insert_query = """
        INSERT INTO public."AssessmentTopic" ("AssessmentCategoryID", "TopicID", "Score")
        VALUES (%s, %s, %s);
    """
    with conn.cursor() as cur:
        for (assessment_category_id, topic_id), score in topic_scores.items():
            if (assessment_category_id, topic_id) in topic_mapping:  
                # If the topic exists, update it
                cur.execute(update_query, (score, topic_mapping[(assessment_category_id, topic_id)]))
            else:
                # If it does not exist, insert a new row
                cur.execute(insert_query, (assessment_category_id, topic_id, score))
        conn.commit()
        print("Topic scores upserted successfully.")
    # Step 4: Fetch updated results
    fetch_query = """
        SELECT * FROM public."AssessmentTopic"
        WHERE "AssessmentCategoryID" IN (
            SELECT "AssessmentCategoryID"
            FROM public."AssessmentCategory"
            WHERE "CompanyAssessmentID" = %(companyassessment_id)s
        );
    """
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"companyassessment_id": companyassessment_id})
        print("Successfully fetched updated topic scores.")
    except Exception as e:
        print(f"Error fetching updated topic scores: {e}")
        results = None
    return results

# Update evaluation type to auto-generated
def rds_update_evaluation_type(conn, companyassessment_id):
    # update evaluation type in the company assessment table
    query = """
            UPDATE public."CompanyAssessment"
            SET "EvaluationTypeID" = 2
            WHERE "CompanyAssessmentID" = %s;
        """
    with conn.cursor() as cursor:
        cursor.execute(query, (companyassessment_id,))
        conn.commit()
    return(f"EvaluationTypeID updated for CompanyAssessmentID: {companyassessment_id}")

def _is_openai_400_bad_request(exc: Exception) -> bool:
    """
    Best-effort detection for OpenAI 400 Bad Request across SDK versions.
    We intentionally avoid importing OpenAI exception types here.
    """
    try:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if status_code == 400:
            return True
        # Some SDKs wrap HTTP response objects
        resp = getattr(exc, "response", None)
        if getattr(resp, "status_code", None) == 400:
            return True
    except Exception:
        pass

    msg = str(exc).lower()
    return ("400" in msg) and ("bad request" in msg or "invalid_request" in msg or "invalid request" in msg)


def _build_openai_fallback_df(questions, score: float = 2.0, error: Exception | None = None) -> pd.DataFrame:
    """
    Create an OpenAI DataFrame compatible with save_to_excel() when OpenAI fails.
    Ensures every question has a row so Excel generation can't break due to missing data.
    """
    # Match extract_openai_responses() behaviour: accept DataFrame or list/iterable of question objects
    if isinstance(questions, pd.DataFrame):
        q_list = list(questions.itertuples(index=False))
    else:
        q_list = list(questions) if questions is not None else []

    err_msg = str(error) if error is not None else "OpenAI request failed"
    response_rows = []
    for index, q in enumerate(q_list, start=1):
        # Best-effort question text extraction (most callsites use q.content)
        question_text = getattr(q, "content", None)
        if question_text is None and isinstance(q, dict):
            question_text = q.get("content") or q.get("question_text") or q.get("question")
        response_rows.append(
            {
                "question_number": index,
                "question_text": question_text,
                "openai_score": float(score),
                "openai_response_text": f"[OpenAI fallback] Defaulted score to {score} due to 400 Bad Request. Error: {err_msg}",
            }
        )
    return pd.DataFrame(response_rows)

# Genai excluded as Openai is far more accurate for emissions extraction purpose
def ai_run_and_save_scores_locally(
    client, 
    #genai_client, 
    path, 
    company_name, 
    instructions, 
    questions, 
    use_chunking=False
    ):
    """
    Local version of the workflow: runs OpenAI and Gemini processing, saves results to an Excel file locally,
    but DOES NOT interact with the database in any way.
    Args:
        client: OpenAI client object.
        genai_client: Gemini client object.
        path: Path to documents (PDF files if use_chunking=False, .txt files if use_chunking=True).
        company_name: Name of the company.
        instructions: Instructions string for Gemini.
        questions: List of questions (prompt objects).
        use_chunking: If True, uses chunking/embedding approach instead of PDF upload.
    """
    print(f"Starting LOCAL process for {company_name} (Chunking: {use_chunking})...")
    
    if use_chunking:
        # Chunking mode: Read .txt files, chunk, create embeddings, and process with chunks
        print("Using chunking/embedding approach for scoring...")
        # Read and clean .txt files
        print(f"Reading and cleaning text files from {path}...")
        all_txts = read_and_clean_txt_files(path)
        if not all_txts:
            print(f"No valid text files found in {path}. Skipping scoring.")
            return
        # Chunk all files
        print("Chunking text...")
        all_chunks = []
        for file in all_txts:
            chunks = chunk_text(file["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "file_name": file["file_name"],
                    "chunk_id": f"{file['file_name']}_chunk_{i+1}",
                    "text": chunk
                })
        print(f"Created {len(all_chunks)} chunks from {len(all_txts)} files")
        # Generate embeddings in batches
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
            batch = all_chunks[i:i+EMBEDDING_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = create_embeddings_batch(client, texts)
            for j, emb in enumerate(embeddings):
                batch[j]["embedding"] = emb
        print("Embeddings created successfully")
        # Process questions with chunk retrieval
        print("Starting parallel execution for OpenAI & Gemini with chunks (local)...")
        with ThreadPoolExecutor() as executor:
            future_openai = executor.submit(
                process_questions_with_chunks,
                client, instructions, questions, all_chunks,
                model=openai_model, top_k=TOP_K_CHUNKS
            )
            # future_gemini = executor.submit(
            #     process_questions_gemini_with_chunks,
            #     genai_client, instructions, questions, all_chunks, client,
            #     model=gemini_model, top_k=TOP_K_CHUNKS
            # )
            openai_df = None
            try:
                openai_results = future_openai.result()
            except Exception as e:
                if _is_openai_400_bad_request(e):
                    print("⚠️ OpenAI returned 400 Bad Request. Defaulting all OpenAI scores to 2 so Excel generation can continue.")
                    openai_df = _build_openai_fallback_df(questions, score=2.0, error=e)
                    openai_results = {}
                else:
                    raise
            #gemini_results = future_gemini.result()
        # Extract responses
        if openai_df is None:
            openai_df = extract_openai_responses_with_chunks(openai_results, questions)
        #gemini_df = extract_gemini_responses_with_chunks(gemini_results, questions)
        gemini_df = None
    else:
        # Original file upload mode
        files = get_files(path)
        print("(Local mode) Getting files for OpenAI and Gemini...")
        # Upload OpenAI files
        file_ids = upload_files_openai(client, path, files)
        # Upload and cache Gemini model safely
        # gemini_cache_name = upload_and_cache_gemini(
        #     genai_client,
        #     path=path,
        #     files=files,
        #     instructions=instructions
        # )
        # if gemini_cache_name is None:
        #     print("⚠️ No valid text content for Gemini cache (all PDFs empty or image-based).")
        #     print("Proceeding without cached content...")
        # Run OpenAI & Gemini processing in parallel (always runs, regardless of cache status)
        print("Starting parallel execution for OpenAI & Gemini (local)...")
        with ThreadPoolExecutor() as executor:
            future_openai = executor.submit(
                process_questions_openai,
                client=client,
                instructions=instructions,
                file_ids=file_ids,
                questions=questions,
                model=openai_model
            )
            # future_gemini = executor.submit(
            #     process_questions_gemini,
            #     genai_client,
            #     gemini_cache_name,  # Can be None
            #     questions
            # )
            openai_df = None
            try:
                openai_results = future_openai.result()
            except Exception as e:
                if _is_openai_400_bad_request(e):
                    print("⚠️ OpenAI returned 400 Bad Request. Defaulting all OpenAI scores to 2 so Excel generation can continue.")
                    openai_df = _build_openai_fallback_df(questions, score=2.0, error=e)
                    openai_results = {}
                else:
                    raise
            #gemini_df = future_gemini.result()
            gemini_df = None
        # Extract OpenAI responses after OpenAI processing is done
        if openai_df is None:
            openai_df = extract_openai_responses(openai_results, questions)
    # Save results to Excel with highlights (NO DB INTERACTIONS)
    print("Saving results locally to Excel (no DB operations)...")
    # Use a special filename to show it's local!
    local_company_name = f"'{company_name}"
    save_to_excel(
        openai_df,
        local_company_name
    )
    print("Local process completed successfully! All data is saved locally (not in the database).")

# (Keeping the original function as is)
def ai_run_and_save_scores_in_rds(conn, client, genai_client, path, company_name, instructions, questions, companyassessment_id, company_secret):
    print(f"Starting process for {company_name}...")
    # Upload OpenAI files
    files = get_files(path)
    print(f"saving files to S3 in company_secret: {company_secret}")
    timestamp = datetime.today().strftime('%Y-%m-%d')
    # for file in files:
    #     file_path = f"{company_name}/{file}"
    #     upload_name = f"""{company_secret} /Metrics 3.0 Documents/{timestamp}/{file}"""
    #     S3_upload_file(file_path, 'prod-esg-scores', upload_name)
    file_ids = upload_files_openai(client, path, files)
    # Upload and cache Gemini model safely
    gemini_cache_name = upload_and_cache_gemini(
        genai_client,
        path=path,
        files=files,
        instructions=instructions
    )
    # Run OpenAI & Gemini processing in parallel (always runs, regardless of cache status)
    print("Starting parallel execution for OpenAI & Gemini...")
    with ThreadPoolExecutor() as executor:
        future_openai = executor.submit(process_questions_openai, client, file_ids, questions, model=openai_model)
        future_gemini = executor.submit(
            process_questions_gemini,
            genai_client,
            gemini_cache_name,  # Can be None
            questions
        )
        openai_results = future_openai.result()
        gemini_df = future_gemini.result()
    # Extract OpenAI responses after OpenAI processing is done
    openai_df = extract_openai_responses(openai_results, questions)
    # Define highlight lists
    gemini_highlight_list = [
        5, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 20, 24, 25, 26, 27, 29, 30, 
        31, 32, 35, 38, 39, 40, 41, 42, 43, 46, 47, 48, 50, 51, 52, 53, 54, 55, 
        56, 57, 58, 59, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 73, 76, 78, 79, 
        80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 98, 99, 100, 101, 
        104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 120
    ]
    openai_highlight_list = [
        1, 2, 3, 4, 8, 9, 12, 19, 21, 22, 23, 28, 33, 34, 36, 37, 44, 45, 
        49, 60, 64, 66, 74, 75, 77, 84, 94, 95, 97, 102, 103, 110, 119
    ]
    # Insert Assessment Questions into the database and get their IDs
    print("Inserting assessment questions into database...")
    try:
        assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)  # Assuming company_assessment_id is available
        print("Assessment questions inserted successfully.")
    except Exception as e:
        print(f"Error inserting assessment questions into database: {e}")
    # Insert OpenAI & Gemini responses into the database
    print("Inserting responses into database...")
    try:
        rds_insert_assessment_responses(conn, questions, openai_df, gemini_df, assessment_question_ids)
        print("Database insertion completed successfully.")
    except Exception as e:
        print(f"Error inserting responses into database: {e}")
    # sleep for 2 seconds to avoid rate limiting
    time.sleep(2)
    rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id)
    # update EvaluationTypeID and SurveyMonkeyModifiedDate in the company assessment table
    query = """
            UPDATE public."CompanyAssessment"
            SET "EvaluationTypeID" = 6, "SurveyMonkeyModifiedDate" = NOW()
            WHERE "CompanyAssessmentID" = %s;
        """
    with conn.cursor() as cursor:
        cursor.execute(query, (companyassessment_id,))
        conn.commit()
    # Save results to Excel with highlights
    save_to_excel(openai_df, gemini_df, company_name, gemini_highlight_list, openai_highlight_list, conn, companyassessment_id)
    print("Updating category scores")
    rds_insert_or_update_category_scores(conn, companyassessment_id)
    rds_insert_or_update_topic_scores(conn, companyassessment_id)
    rds_update_survey_score(conn, companyassessment_id)
    print("Process completed successfully!")

# -------------------------
# Chunking and Embedding Functions for L1+ and L2 Scoring
# -------------------------
# Configuration constants
CHUNK_SIZE = 1000  # approximate tokens per chunk
CHUNK_OVERLAP = 100  # overlapping tokens between chunks
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 10  # number of chunks to send per API call
TOP_K_CHUNKS = 5  # retrieve top 5 chunks per question
CHUNK_SNIPPET_LENGTH = 500  # characters to save per chunk for auditing

def read_and_clean_txt_files(path):
    """
    Read and clean .txt files from a folder.
    Uses kadoa_functions for cleaning if available, otherwise basic cleaning.
    :param folder_path: Path to folder containing .txt files
    :return: List of dicts with 'file_name' and 'text' keys
    """
    texts = []
    try:
        import kadoa_functions as kadoa
        use_kadoa_cleaning = True
    except ImportError:
        use_kadoa_cleaning = False
    p = Path(path)

    try:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()

        if use_kadoa_cleaning:
            # Use kadoa cleaning functions
            text = kadoa.clean_html_file(str(p))
            text = kadoa.filter_out_cookie_lines(text)
            # This preserves the original line breaks and tables
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            # text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
        else:
            # Basic cleaning with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator="\n")
            # Filter cookie lines
            keywords = ["cookie", "cookies", "tracking", "consent", "privacy policy"]
            lines = text.splitlines()
            filtered_lines = [
                line for line in lines
                if not any(keyword in line.lower() for keyword in keywords)
            ]
            # This preserves the original line breaks and tables
            text = "\n".join(line.strip() for line in filtered_lines if line.strip())
        
        if text and text.strip():
            texts.append({
                "file_name": p.name,
                "text": text
            })
    except Exception as e:
        print(f"Error reading {p.name}: {e}")
    return texts

def chunk_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Chunk text but keep Markdown tables as atomic chunks.
    :param text: Text to chunk
    :param max_tokens: Approximate tokens per chunk (using word count as proxy)
    :param overlap: Overlapping tokens between chunks
    :return: List of chunk strings
    """
    lines = text.splitlines()
    blocks = []
    buffer = []
    inside_table = False

    # ---- STEP 1: split into blocks (tables vs normal text) ----
    for line in lines:
        if line.strip().startswith("|"):
            if not inside_table:
                if buffer:
                    blocks.append("\n".join(buffer))
                    buffer = []
                inside_table = True
            buffer.append(line)
        else:
            if inside_table:
                blocks.append("\n".join(buffer))
                buffer = []
                inside_table = False
            buffer.append(line)

    if buffer:
        blocks.append("\n".join(buffer))

    # ---- STEP 2: chunk only non-table blocks ----
    chunks = []

    for block in blocks:
        if block.strip().startswith("|"):
            # table → atomic chunk
            chunks.append(block)
        else:
            words = block.split()
            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                chunk_words = words[start:end]
                chunks.append(" ".join(chunk_words))
                start += max_tokens - overlap

    return chunks

    # Previous chunking logic
    '''
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += max_tokens - overlap
    return chunks
    '''

def create_embeddings_batch(client, text_list, model=EMBEDDING_MODEL):
    """
    Send multiple chunks in a single API call to create embeddings.
    :param client: OpenAI client
    :param text_list: List of text strings to embed
    :param model: Embedding model name
    :return: List of embedding vectors
    """
    resp = client.embeddings.create(
        model=model,
        input=text_list
    )
    return [item.embedding for item in resp.data]

    # embeddings = []

    # for i, text in enumerate(text_list):
    #     tokens = len(ENCODING.encode(text))
    #     if tokens > 8000:
    #         print("❌ PRE-FLIGHT TOKEN VIOLATION")
    #         print(f"Index: {i} | Tokens: {tokens}")
    #         print("----- CHUNK START -----")
    #         print(text)
    #         print("----- CHUNK END -----")
    #         raise ValueError("Chunk too large")

    #     try:
    #         resp = client.embeddings.create(
    #             model=model,
    #             input=text
    #         )
    #         embeddings.append(resp.data[0].embedding)

    #     except Exception as e:
    #         print("❌ EMBEDDING FAILED FOR SINGLE CHUNK")
    #         print(f"Index: {i}")
    #         print(f"Token count: {tokens}")
    #         print("----- CHUNK START -----")
    #         print(text)
    #         print("----- CHUNK END -----")
    #         raise

    # return embeddings

def cosine_similarity_numpy(a, b):
    """
    Calculate cosine similarity between vector a and matrix b.
    :param a: Single embedding vector (1D array)
    :param b: Matrix of embedding vectors (2D array)
    :return: Array of similarity scores
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sims = np.dot(b_norm, a_norm)
    return sims

def retrieve_top_chunks(question_embedding, all_chunks, top_k=TOP_K_CHUNKS):
    """
    Retrieve top-k most similar chunks for a question embedding.
    :param question_embedding: Embedding vector for the question
    :param all_chunks: List of chunk dicts with 'text' and 'embedding' keys
    :param top_k: Number of top chunks to retrieve
    :return: List of top-k chunk dicts
    """
    chunk_embeddings = np.array([c["embedding"] for c in all_chunks])
    sims = cosine_similarity_numpy(np.array(question_embedding), chunk_embeddings)
    top_indices = sims.argsort()[::-1][:top_k]
    return [all_chunks[i] for i in top_indices]

def process_questions_with_chunks(client, instructions, questions, all_chunks,
                                  model=openai_model, retries=3, retry_delay=3,
                                  max_workers=5, top_k=TOP_K_CHUNKS, snippet_length=CHUNK_SNIPPET_LENGTH):
    """
    Process questions using chunk retrieval (semantic search).
    :param client: OpenAI client
    :param instructions: Instructions for scoring
    :param questions: List of question objects (or DataFrame)
    :param all_chunks: List of chunk dicts with 'text' and 'embedding' keys
    :param model: OpenAI model name
    :param retries: Number of retry attempts
    :param retry_delay: Delay between retries (seconds)
    :param max_workers: Max parallel workers
    :param top_k: Number of top chunks to retrieve per question
    :param snippet_length: Characters to save per chunk for auditing
    :return: Dict mapping question.label to response dict with 'response', 'chunks', 'full_input_text'
    """
    results = {}
    # Handle DataFrame input
    if isinstance(questions, pd.DataFrame):
        if questions.empty:
            print("No questions provided, skipping processing.")
            return results
        questions = list(questions.itertuples(index=False))
    else:
        if not questions or len(questions) == 0:
            print("No questions provided, skipping processing.")
            return results
    total_questions = len(questions)
    print(f"Processing {total_questions} questions using chunk retrieval...")
    def process_single_question(index, question):
        for attempt in range(retries):
            try:
                print(f"({index}/{total_questions}) Processing: {question.label} (Attempt {attempt + 1})")
                # 1∩╕ÅΓâú Embed the question
                q_emb_resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=question.content
                )
                question_embedding = q_emb_resp.data[0].embedding
                # 2∩╕ÅΓâú Retrieve top chunks
                top_chunks = retrieve_top_chunks(question_embedding, all_chunks, top_k=top_k)
                retrieved_text = "\n\n".join([c["text"] for c in top_chunks])
                # Save snippet of each chunk for auditing
                chunk_snippets = [c["text"][:snippet_length].replace("\n", " ") for c in top_chunks]
                # 3∩╕ÅΓâú Build GPT input with retrieved context
                full_input_text = f"{instructions}\n\nContext:\n{retrieved_text}\n\nQuestion: {question.content}"
                # 4∩╕ÅΓâú Call GPT (using responses API, no file uploads needed)
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": full_input_text}]}]
                )
                # 5∩╕ÅΓâú Return GPT response + chunks + full input text
                return {
                    "response": resp,
                    "chunks": chunk_snippets,
                    "full_input_text": full_input_text
                }
            except Exception as e:
                print(f"Error processing {question.label}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        return None
    # Multithreaded execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_question = {
            executor.submit(process_single_question, idx, question): question
            for idx, question in enumerate(questions, start=1)
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            resp_obj = future.result()
            if resp_obj is not None:
                results[question.label] = resp_obj
    print("Processing complete.")
    return results

def extract_openai_responses_with_chunks(results, questions, snippet_length=CHUNK_SNIPPET_LENGTH):
    """
    Extract OpenAI responses into a DataFrame with chunk information.
    :param results: Dict from process_questions_with_chunks
    :param questions: List of question objects
    :param snippet_length: Characters to save per chunk snippet
    :return: DataFrame with question_number, question_text, openai_score, openai_response_text, chunks_used
    """
    response_rows = []
    if isinstance(questions, pd.DataFrame):
        questions = list(questions.itertuples(index=False))
    def _extract_output_text(resp_obj):
        resp = resp_obj.get("response") if isinstance(resp_obj, dict) else resp_obj
        if resp is None:
            return ""
        # Try simple output_text first
        output_text_candidates = getattr(resp, "output_text", None)
        if isinstance(output_text_candidates, list):
            for candidate in output_text_candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        # Fall back to structured output
        outputs = getattr(resp, "output", None) or []
        for entry in outputs:
            content_items = entry.get("content") if isinstance(entry, dict) else getattr(entry, "content", None)
            if not content_items:
                continue
            if not isinstance(content_items, list):
                content_items = [content_items]
            for item in content_items:
                text_value = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if text_value:
                    return text_value.strip()
        resp_text = getattr(resp, "text", None)
        if isinstance(resp_text, str):
            return resp_text.strip()
        return ""
    for index, question in enumerate(questions, start=1):
        resp_obj = results.get(question.label)
        if not resp_obj:
            continue
        output_text = _extract_output_text(resp_obj)
        score = 0
        try:
            if output_text:
                last_word = output_text.split(" ")[-1]
                match = re.match(r"\d+(\.\d+)?$", last_word)
                if match:
                    score = float(match.group())
        except:
            pass
        # Build readable chunks: remove duplicates, truncate to snippet_length
        raw_chunks = resp_obj.get("chunks", [])
        unique_chunks = []
        seen = set()
        for c in raw_chunks:
            chunk_snip = c[:snippet_length].replace("\n", " ").strip()
            if chunk_snip and chunk_snip not in seen:
                unique_chunks.append(chunk_snip)
                seen.add(chunk_snip)
        response_rows.append({
            "question_number": index,
            "question_text": question.content,
            "openai_score": score,
            "openai_response_text": output_text,
            "chunks_used": " | ".join(unique_chunks)
        })
    return pd.DataFrame(response_rows)

def process_questions_gemini_with_chunks(
    genai_client,
    instructions,
    questions,
    all_chunks,
    openai_client,
    model=gemini_model,
    retries=3,
    retry_delay=3,
    max_workers=5,
    top_k=TOP_K_CHUNKS,
    snippet_length=CHUNK_SNIPPET_LENGTH
):
    """
    Process questions using chunk retrieval for Gemini.
    :param genai_client: Gemini client object
    :param instructions: Instructions for scoring
    :param questions: List of question objects (or DataFrame)
    :param all_chunks: List of chunk dicts with 'text' and 'embedding' keys
    :param openai_client: OpenAI client for creating question embeddings
    :param model: Gemini model name
    :param retries: Number of retry attempts
    :param retry_delay: Delay between retries (seconds)
    :param max_workers: Max parallel workers
    :param top_k: Number of top chunks to retrieve per question
    :param snippet_length: Characters to save per chunk for auditing
    :return: Dict mapping question.label to response dict with 'response', 'chunks', 'full_input_text'
    """
    results = {}
    # Handle DataFrame input
    if isinstance(questions, pd.DataFrame):
        if questions.empty:
            print("No questions provided, skipping Gemini processing.")
            return results
        questions = list(questions.itertuples(index=False))
    else:
        if not questions or len(questions) == 0:
            print("No questions provided, skipping Gemini processing.")
            return results
    total_questions = len(questions)
    print(f"Processing {total_questions} questions using chunk retrieval with Gemini...")
    def process_single_question(index, question):
        for attempt in range(retries):
            try:
                print(f"({index}/{total_questions}) Processing Gemini: {question.label} (Attempt {attempt + 1})")
                # 1. Embed the question using OpenAI embeddings
                q_emb_resp = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=question.content
                )
                question_embedding = q_emb_resp.data[0].embedding
                # 2. Retrieve top chunks
                top_chunks = retrieve_top_chunks(question_embedding, all_chunks, top_k=top_k)
                retrieved_text = "\n\n".join([c["text"] for c in top_chunks])
                # Save snippet of each chunk for auditing
                chunk_snippets = [c["text"][:snippet_length].replace("\n", " ") for c in top_chunks]
                '''
                # 3. Build Gemini input with retrieved context
                full_input_text = f"{instructions}\n\nContext:\n{retrieved_text}\n\nQuestion: {question.content}"
                # 4. Call Gemini (no cached content, chunks are in the prompt)
                response = genai_client.models.generate_content(
                    model=model,
                    contents=[full_input_text],
                )
                '''
                response = genai_client.models.generate_content(
                    model=model,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": instructions},
                                {"text": "\n\nContext:\n" + retrieved_text},
                                {"text": "\n\nQuestion:\n" + question.content}
                            ]
                        }
                    ],
                )
                # Keep this for auditing
                full_input_text = f"{instructions}\n\nContext:\n{retrieved_text}\n\nQuestion: {question.content}"
                # 5. Return Gemini response + chunks + full input text
                return {
                    "response": response,
                    "chunks": chunk_snippets,
                    "full_input_text": full_input_text
                }
            except Exception as e:
                print(f"Error processing Gemini question {question.label}: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        return None
    # Multithreaded execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_question = {
            executor.submit(process_single_question, idx, question): question
            for idx, question in enumerate(questions, start=1)
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            resp_obj = future.result()
            if resp_obj is not None:
                results[question.label] = resp_obj
    print("Gemini chunking processing complete.")
    return results

def extract_gemini_responses_with_chunks(results, questions, snippet_length=CHUNK_SNIPPET_LENGTH):
    """
    Extract Gemini responses into a DataFrame with chunk information.
    :param results: Dict from process_questions_gemini_with_chunks
    :param questions: List of question objects
    :param snippet_length: Characters to save per chunk snippet
    :return: DataFrame with question_number, question_text, gemini_score, gemini_response_text, chunks_used
    """
    response_rows = []
    if isinstance(questions, pd.DataFrame):
        questions = list(questions.itertuples(index=False))
    def _extract_output_text(resp_obj):
        resp = resp_obj.get("response") if isinstance(resp_obj, dict) else resp_obj
        if resp is None:
            return ""
        # Extract text from Gemini response
        text = getattr(resp, "text", None) or getattr(resp, "content", "")
        if isinstance(text, str):
            return text.strip()
        return ""
    for index, question in enumerate(questions, start=1):
        resp_obj = results.get(question.label)
        if not resp_obj:
            continue
        output_text = _extract_output_text(resp_obj)
        score = 0
        try:
            if output_text:
                last_word = output_text.split(" ")[-1]
                match = re.match(r"\d+(\.\d+)?$", last_word)
                if match:
                    score = float(match.group())
        except:
            pass
        # Build readable chunks: remove duplicates, truncate to snippet_length
        raw_chunks = resp_obj.get("chunks", [])
        unique_chunks = []
        seen = set()
        for c in raw_chunks:
            chunk_snip = c[:snippet_length].replace("\n", " ").strip()
            if chunk_snip and chunk_snip not in seen:
                unique_chunks.append(chunk_snip)
                seen.add(chunk_snip)
        response_rows.append({
            "question_number": index,
            "question_text": question.content,
            "gemini_score": score,
            "gemini_response_text": output_text,
            "chunks_used": " | ".join(unique_chunks)
        })
    return pd.DataFrame(response_rows)

# -------------------------
def generate_m3_score(conn, companyassessment_id, clean_pdfs_path, secrets, scoring_level='L2', company_name=None, use_chunking=False):
    """
    Generate an M3 score for a given CompanyAssessmentID.
    :param conn: psycopg2 database connection object.
    :param companyassessment_id: ID of the CompanyAssessment.
    :param clean_pdfs_path: Local path to cleaned PDF files (or .txt files if use_chunking=True).
    :param scoring_level: 'L1+' (OpenAI only, L1+ survey 1111),
                          'L2' (OpenAI only, survey 3333),
                          or 'L3' (OpenAI + Gemini, survey 3333).
    :param secrets: Dictionary containing necessary secrets like API keys from secrets manager.
    :param company_name: Company name (optional, will be fetched from DB if not provided).
    :param use_chunking: If True, uses chunking/embedding approach instead of PDF upload.
                        Works for L1+, L2, and L3. When False, L3 uses Gemini cached content.
    :return: Dictionary with CompanyAssessmentID, or None on error/invalid level.
    """
    print(f"Generating M3 score for CompanyAssessmentID {companyassessment_id} (Level: {scoring_level}, Chunking: {use_chunking})")
    if scoring_level == 'L1+':
        questions = rds_get_prompts(conn, survey_id=1111)
    elif scoring_level in ('L2', 'L3'):
        questions = rds_get_prompts(conn)
    else:
        print(f"Invalid scoring level: {scoring_level}")
        return
    # If company_name is not provided, fetch it from the database
    if not company_name:
        query = """
            SELECT "CompanyName" FROM public."CompanyAssessment" 
            LEFT JOIN public."Company" ON "CompanyAssessment"."CompanyID" = "Company"."CompanyID"
            WHERE "CompanyAssessmentID" = %s;
        """
        with conn.cursor() as cur:
            cur.execute(query, (companyassessment_id,))
            result = cur.fetchone()
            if result:
                company_name = result[0]
            else:
                print(f"No company name found for CompanyAssessmentID {companyassessment_id}.")
                return None
    # Initialize OpenAI client  - this is common in all supported scoring levels
    client = OpenAI(
    organization=secrets['openai_organization'],
    project=secrets['openai_project'],
    api_key=secrets['open_api_key']
    )
    # Run gemini API if scoring level is L3
    if scoring_level == 'L3':
        gemini_api_key = secrets['gemini_api_key']
        genai_client = genai.Client(api_key=gemini_api_key)
    # Handle chunking mode for L1+, L2, and L3
    if use_chunking and scoring_level in ('L1+', 'L2', 'L3'):
        print("≡ƒôÜ Using chunking/embedding approach for scoring...")
        # Read and clean .txt files
        print(f"≡ƒôû Reading and cleaning text files from {clean_pdfs_path}...")
        all_txts = read_and_clean_txt_files(clean_pdfs_path)
        if not all_txts:
            print(f"Γ¥î No valid text files found in {clean_pdfs_path}. Skipping scoring.")
            return None
        # Chunk all files
        print("Γ£é∩╕Å Chunking text...")
        all_chunks = []
        for file in all_txts:
            chunks = chunk_text(file["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "file_name": file["file_name"],
                    "chunk_id": f"{file['file_name']}_chunk_{i+1}",
                    "text": chunk
                })
        print(f"Γ£à Created {len(all_chunks)} chunks from {len(all_txts)} files")
        # Generate embeddings in batches
        print(f"≡ƒºá Creating embeddings for {len(all_chunks)} chunks...")
        for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
            batch = all_chunks[i:i+EMBEDDING_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = create_embeddings_batch(client, texts)
            for j, emb in enumerate(embeddings):
                batch[j]["embedding"] = emb
        print("Γ£à Embeddings created successfully")
        # Process questions with chunk retrieval
        if scoring_level == 'L1+':
            print("Starting L1+ processing with chunk retrieval...")
            openai_results = process_questions_with_chunks(
                client, instructions, questions, all_chunks,
                model=openai_model, top_k=TOP_K_CHUNKS
            )
            openai_df = extract_openai_responses_with_chunks(openai_results, questions)
            # Insert prompts into database
            try:
                print("Inserting assessment questions into database...")
                assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
                print("Assessment questions inserted.")
            except Exception as e:
                print(f"Error inserting assessment questions: {e}")
                return
            # Insert OpenAI responses
            try:
                print("Inserting OpenAI responses...")
                rds_insert_assessment_responses(
                    conn,
                    questions,
                    openai_df=openai_df,
                    assessment_question_ids=assessment_question_ids,
                    engines=('openai',)
                )
                print("OpenAI responses inserted.")
            except Exception as e:
                print(f"Error inserting OpenAI responses: {e}")
                return
            time.sleep(2)  # Avoid rate limiting
            # Update answers and scores from OpenAI responses
            rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id, engine_id=1)
            # Update category and topic scores
            print("Updating category scores...")
            rds_insert_or_update_category_scores(conn, companyassessment_id)
            print("Updating topic scores...")
            rds_insert_or_update_topic_scores(conn, companyassessment_id)
            print("Updating survey score...")
            rds_update_survey_score(conn, companyassessment_id)
        elif scoring_level == 'L2':
            print("Starting L2 processing with chunk retrieval...")
            openai_results = process_questions_with_chunks(
                client, instructions, questions, all_chunks,
                model=openai_model, top_k=TOP_K_CHUNKS
            )
            openai_df = extract_openai_responses_with_chunks(openai_results, questions)
            # Insert prompts into database
            try:
                print("Inserting assessment questions into database...")
                assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
                print("Assessment questions inserted.")
            except Exception as e:
                print(f"Error inserting assessment questions: {e}")
                return
            # Insert OpenAI responses
            try:
                print("Inserting OpenAI responses...")
                rds_insert_assessment_responses(
                    conn,
                    questions,
                    openai_df=openai_df,
                    assessment_question_ids=assessment_question_ids,
                    engines=('openai',)
                )
                print("OpenAI responses inserted.")
            except Exception as e:
                print(f"Error inserting OpenAI responses: {e}")
                return
            time.sleep(2)  # Avoid rate limiting
            # Update answers and scores from OpenAI responses
            rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id, engine_id=1)
            # Update category and topic scores
            print("Updating category scores...")
            rds_insert_or_update_category_scores(conn, companyassessment_id)
            print("Updating topic scores...")
            rds_insert_or_update_topic_scores(conn, companyassessment_id)
            print("Updating survey score...")
            rds_update_survey_score(conn, companyassessment_id)
        elif scoring_level == 'L3':
            print("Starting L3 processing with chunk retrieval...")
            # Process OpenAI questions with chunks
            openai_results = process_questions_with_chunks(
                client, instructions, questions, all_chunks,
                model=openai_model, top_k=TOP_K_CHUNKS
            )
            openai_df = extract_openai_responses_with_chunks(openai_results, questions)
            # Process Gemini questions with chunks
            gemini_results = process_questions_gemini_with_chunks(
                genai_client, instructions, questions, all_chunks, client,
                model=gemini_model, top_k=TOP_K_CHUNKS
            )
            gemini_df = extract_gemini_responses_with_chunks(gemini_results, questions)
            # Insert prompts into database
            try:
                print("Inserting assessment questions into database...")
                assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
                print("Assessment questions inserted.")
            except Exception as e:
                print(f"Error inserting assessment questions: {e}")
                return
            # Insert both OpenAI and Gemini responses
            try:
                print("Inserting OpenAI and Gemini responses...")
                rds_insert_assessment_responses(
                    conn,
                    questions,
                    openai_df=openai_df,
                    gemini_df=gemini_df,
                    assessment_question_ids=assessment_question_ids
                )
                print("Responses inserted.")
            except Exception as e:
                print(f"Error inserting responses: {e}")
                return
            time.sleep(2)  # Avoid rate limiting
            # Update answer/score from whichever response was marked as 'Used'
            rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id)
            # Update category and topic scores
            print("Updating category scores...")
            rds_insert_or_update_category_scores(conn, companyassessment_id)
            print("Updating topic scores...")
            rds_insert_or_update_topic_scores(conn, companyassessment_id)
            print("Updating survey score...")
            rds_update_survey_score(conn, companyassessment_id)
        return {"CompanyAssessmentID": companyassessment_id}
    # Original file upload approach (for L3 or when use_chunking=False)
    # Upload files 
    files = get_files(clean_pdfs_path)
    file_ids = upload_files_openai(client, clean_pdfs_path, files)
    if scoring_level == 'L1+':
        # --- L1+: OpenAI-only processing ---
        print("Starting OpenAI processing...")
        openai_results = process_questions_openai(client, instructions, file_ids, questions, model = openai_model)
        openai_df = extract_openai_responses(openai_results, questions)
        print(openai_df)
        # Insert prompts into database
        try:
            print("Inserting assessment questions into database...")
            assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
            print("Assessment questions inserted.")
        except Exception as e:
            print(f"Error inserting assessment questions: {e}")
            return
        # Insert OpenAI responses
        try:
            print("Inserting OpenAI responses...")
            rds_insert_assessment_responses(
                conn,
                questions,
                openai_df=openai_df,
                assessment_question_ids=assessment_question_ids,
                engines=('openai',)
            )
            print("OpenAI responses inserted.")
        except Exception as e:
            print(f"Error inserting OpenAI responses: {e}")
            return
        time.sleep(2)  # Avoid rate limiting
        # Update answers and scores from OpenAI responses
        rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id, engine_id=1)
    elif scoring_level == 'L2':
        # --- L2: OpenAI-only processing ---
        print("Starting OpenAI processing...")
        openai_results = process_questions_openai(client, instructions, file_ids, questions,model = openai_model)
        openai_df = extract_openai_responses(openai_results, questions)
        # Insert prompts into database
        try:
            print("Inserting assessment questions into database...")
            assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
            print("Assessment questions inserted.")
        except Exception as e:
            print(f"Error inserting assessment questions: {e}")
            return
        # Insert OpenAI responses
        try:
            print("Inserting OpenAI responses...")
            rds_insert_assessment_responses(
                conn,
                questions,
                openai_df=openai_df,
                assessment_question_ids=assessment_question_ids,
                engines=('openai',)
            )
            print("OpenAI responses inserted.")
        except Exception as e:
            print(f"Error inserting OpenAI responses: {e}")
            return
        time.sleep(2)  # Avoid rate limiting
        # Update answers and scores from OpenAI responses
        rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id, engine_id=1)
    elif scoring_level == 'L3':
        # --- L3: OpenAI + Gemini processing ---
        print("Starting OpenAI and Gemini processing...")
        # Upload and cache Gemini model
        gemini_cache_name = upload_and_cache_gemini(genai_client, clean_pdfs_path, files, instructions)
        # Run OpenAI and Gemini in parallel
        with ThreadPoolExecutor() as executor:
            future_openai = executor.submit(process_questions_openai, client, instructions, file_ids, questions, model = openai_model)
            future_gemini = executor.submit(process_questions_gemini, genai_client, gemini_cache_name, questions)
            openai_results = future_openai.result()
            gemini_df = future_gemini.result()
        openai_df = extract_openai_responses(openai_results, questions)
        # Insert prompts into database
        try:
            print("Inserting assessment questions into database...")
            assessment_question_ids = rds_insert_into_assessmentquestion(conn, questions, companyassessment_id)
            print("Assessment questions inserted.")
        except Exception as e:
            print(f"Error inserting assessment questions: {e}")
            return
        # Insert both OpenAI and Gemini responses
        try:
            print("Inserting OpenAI and Gemini responses...")
            rds_insert_assessment_responses(
                conn,
                questions,
                openai_df=openai_df,
                gemini_df=gemini_df,
                assessment_question_ids=assessment_question_ids
            )
            print("Responses inserted.")
        except Exception as e:
            print(f"Error inserting responses: {e}")
            return
        time.sleep(2)
        # Update answer/score from whichever response was marked as 'Used'
        rds_update_assessmentquestions_with_used_responses(conn, companyassessment_id)
    else:
        print(f"Invalid scoring level: {scoring_level}")
        return

# --- Function to download S3 pages ---
def download_pages_from_s3(s3_folder_link, local_folder_name, secrets):
    import urllib.parse
    # Extract bucket and prefix from console-style link
    parsed = urllib.parse.urlparse(s3_folder_link)
    query_params = urllib.parse.parse_qs(parsed.query)
    bucket_name = parsed.path.split('/')[-1]  # gets 'prod-webscraped-text'
    prefix = query_params.get('prefix', [''])[0]  # gets the folder prefix
    region = query_params.get('region', ['eu-west-2'])[0]
    print(f"Parsed bucket: {bucket_name}")
    print(f"Parsed prefix: {prefix}")
    s3 = boto3.client(
        's3',
        aws_access_key_id=secrets.get("api_access_key_id"),
        aws_secret_access_key=secrets.get("api_secret_access_key"),
        region_name="eu-west-2"
    )
    local_path = local_folder_name
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.txt'):
                filename = os.path.basename(key)
                MAX_FILENAME_LEN = 150
                if len(filename) > MAX_FILENAME_LEN:
                    name_part, ext = os.path.splitext(filename)
                    name_part = name_part[:MAX_FILENAME_LEN - len(ext)]
                    filename = name_part + ext
                local_file_path = os.path.join(local_path, filename)
                s3.download_file(bucket_name, key, local_file_path)
                print(f"Downloaded {key} to {local_file_path}")
    return local_path
