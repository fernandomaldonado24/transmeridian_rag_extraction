# Load required libraries
import pandas as pd
from collections import namedtuple
from collections import defaultdict
import openai
import os
from datetime import datetime
import json
import openpyxl
import importlib
import esg_ai_pipeline_functions as esg
importlib.reload(esg)
import paramiko
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import psycopg2
import logging
import uuid
import re
import numpy as np
import boto3
import botocore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from botocore.exceptions import ClientError
from email.mime.image import MIMEImage
import math
import kadoa_functions as kadoa
importlib.reload(kadoa)
import time
from pathlib import Path
import psycopg2.extras
from urllib.parse import urlparse, urljoin, quote
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, unquote
import gc
import shutil

######## general data declarations ########

# These are the keywords we will use to identify relevant documents during the scrape so we can download them and upload to S3.
tier1_keywords = ['annual', 'sustainability', 'impact', 'emissions', 'gender', 'responsible', 
                  'climate', 'TCFD', 'transparency', 'stewardship', 'esg']

tier2_keywords = ['environmental', 'social', 'GRI', 'SASB', 'carbon', 'conduct', 'diversity', 
                  'pay', 'tax', 'risk', 'human', 'health', 'task force', 'zero', 'strategy']

tier3_keywords = ['ISO', 'anti', 'bribery', 'corruption', 'slavery', 'whistleblowing', 
                  'minerals', 'equal']


######## DOWNLOADING LINKS FUNCTIONS ########

def sanitize_filename(name: str) -> str:
    """Remove invalid characters for file names."""
    return re.sub(r'[^a-zA-Z0-9_\-\. ]', '_', name).strip()

def filename_from_url(url: str) -> str:
    """
    Extract the filename from the URL
    """
    path = urlparse(url).path             # '/wp-content/uploads/.../file.pdf'
    name = os.path.basename(path)         # 'file.pdf'
    return unquote(name)                  # handle URL-encoded characters

def download_pdfs(links, save_dir, company):
    os.makedirs(save_dir, exist_ok=True)
    downloaded_files = []

    for i, link in enumerate(links, start=1):
        url = link[0] if isinstance(link, tuple) else link

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Use filename from URL
            filename = filename_from_url(url)
            filename = sanitize_filename(filename)  # in case of unsafe characters

            final_path = os.path.join(save_dir, filename)
            with open(final_path, "wb") as f:
                f.write(response.content)

            downloaded_files.append(final_path)
            print(f"    Γ£ô Saved: {filename}")
        except Exception as e:
            print(f"    Γ£ù Failed: {url} ({e})")

    return downloaded_files

def find_terms_in_links(doc_links, terms_list):
    """
    Count occurrences of terms in filtered PDF links.
    Returns dict with term -> list of matching links
    """
    results = defaultdict(list)
    terms_lower = [term.lower() for term in terms_list]

    for link in doc_links:
        link_for_matching = link.lower()
        for i, term_lower in enumerate(terms_lower):
            if term_lower in link_for_matching:
                results[terms_list[i]].append(link)
    
    return dict(results)


def extract_year_from_link(link):
    """Extract the most recent year from a PDF link filename"""
    parsed = urlparse(link)
    filename = os.path.basename(parsed.path)
    
    # Look for 4-digit years (2000-2099)
    four_digit_years = [int(y) for y in re.findall(r"(20\d{2})", filename)]
    
    # Look for 2-digit years (assuming 20xx format: FY24, 24, etc.)
    two_digit_years = [int(y) for y in re.findall(r"(?:FY|fy)?(\d{2})", filename)]
    # Only keep reasonable 2-digit years and convert to 20xx
    two_digit_years = [2000 + y for y in two_digit_years if 20 <= y <= 35]
    
    all_years = four_digit_years + two_digit_years
    
    if not all_years:
        return None
    
    return max(all_years)  # return the most recent year

def get_best_document_for_term(links, term):
    """
    Select the best (most recent) document for a given term.
    Returns the single best link or None if no valid links.
    """
    if not links:
        return None
    
    # Group links by year
    links_by_year = defaultdict(list)
    links_without_year = []
    
    for link in links:
        year = extract_year_from_link(link)
        if year:
            links_by_year[year].append(link)
        else:
            links_without_year.append(link)
    
    # Select from the most recent year if available
    if links_by_year:
        most_recent_year = max(links_by_year.keys())
        best_link = links_by_year[most_recent_year][0]  # Take first if multiple in same year
        
        # Show what we're filtering out
        older_years = [year for year in links_by_year.keys() if year < most_recent_year]
        if older_years:
            print(f"    Filtered out older {term} documents from years: {sorted(older_years)}")
        
        return best_link
    
    # If no year info, just take the first one
    return links_without_year[0] if links_without_year else None

def select_documents_by_tiers(doc_links, tier1_terms, tier2_terms, tier3_terms, max_docs=5):
    """Select up to max_docs documents using tiered keyword approach."""
    selected_links = []
    selected_set = set()
    used_terms = set()

    # Tier 1
    tier1_results = find_terms_in_links(doc_links, tier1_terms)
    for term, links in tier1_results.items():
        if term not in used_terms and len(selected_links) < max_docs:
            best_link = get_best_document_for_term(links, term)
            if best_link and best_link not in selected_set:
                selected_links.append((best_link, term))
                selected_set.add(best_link)
                used_terms.add(term)

    # Tier 2
    tier2_results = find_terms_in_links(doc_links, tier2_terms)
    for term, links in tier2_results.items():
        if term not in used_terms and len(selected_links) < max_docs:
            best_link = get_best_document_for_term(links, term)
            if best_link and best_link not in selected_set:
                selected_links.append((best_link, term))
                selected_set.add(best_link)
                used_terms.add(term)

    # Tier 3
    tier3_results = find_terms_in_links(doc_links, tier3_terms)
    for term, links in tier3_results.items():
        if term not in used_terms and len(selected_links) < max_docs:
            best_link = get_best_document_for_term(links, term)
            if best_link and best_link not in selected_set:
                selected_links.append((best_link, term))
                selected_set.add(best_link)
                used_terms.add(term)

    return selected_links


def url_exists(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=5)
        return response.status_code not in [404, 403]
    except requests.RequestException:
        return False

def normalize_pdf_link(href, base_url=None):
    href = href.strip()
    if base_url and not href.startswith(('http://', 'https://')):
        href = urljoin(base_url, href)
    parsed = urlparse(href)
    path_encoded = quote(parsed.path)
    normalized = parsed._replace(path=path_encoded).geturl()
    return normalized

def filter_links(doc_links, year_cutoff=2022):
    """
    Apply global filtering:
    - Deduplicate (already a set)
    - Remove 404/403
    - Keep only links with year >= cutoff (based on filename, not folder path)
    """
    valid_links = []
    
    for link in doc_links:
        parsed = urlparse(link)
        filename = os.path.basename(parsed.path)  # only the file name (e.g., "Annual_Report_2015.pdf")

        # Year check: look for a 4-digit year in the filename only
        years = re.findall(r"(20\d{2})", filename)
        if years and max(map(int, years)) < year_cutoff:
            continue  # skip pre-cutoff links

        if url_exists(link):
            valid_links.append(link)
        else:
            print(f"404 detected, removing link: {link}")

    return valid_links

def find_pdf_links_in_folder(folder_path, base_url=None):
    doc_links = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    soup = BeautifulSoup(content, 'html.parser')
                    links = soup.find_all('a', href=True)

                    for link in links:
                        href = link['href']
                        try:
                            # Safely attempt to parse the URL
                            parsed = urlparse(href)
                            if parsed.path.lower().endswith('.pdf'):
                                normalized = normalize_pdf_link(href, base_url)
                                doc_links.add(normalized)
                        except ValueError as e:
                            # Log the error and skip the invalid link
                            print(f"Skipping invalid URL: {href} due to error: {e}")
                            continue  # Move to the next link
    return list(doc_links)

def download_company_documents(path_to_webpages, company_name, company_secret, company_url, env, secrets):
    year_cutoff = 2022
    max_documents_per_company = 10
    all_selected_documents = {}

    # Step 1: find links
    doc_links = find_pdf_links_in_folder(path_to_webpages, company_url)
    print(f"  Found {len(doc_links)} PDF links")
    
    # Step 2: filter
    doc_links = filter_links(doc_links, year_cutoff)
    print(f"  After filtering: {len(doc_links)} valid links")

    # Step 3: select by tier
    selected_docs = select_documents_by_tiers(
        doc_links, 
        tier1_keywords, 
        tier2_keywords, 
        tier3_keywords, 
        max_documents_per_company
    )
    
    all_selected_documents[company_name] = selected_docs
    print(f"Final selection: {len(selected_docs)} documents")
    
    # Step 4: download into /tmp
    base_dir = Path("/tmp")
    save_dir = os.path.join(base_dir, "Company Documents", company_name)
    # ensure directory exists
    os.makedirs(save_dir, exist_ok=True)


    downloaded_files = download_pdfs(selected_docs, save_dir, company_name)

    # Step 5: upload to S3 
    s3_bucket = f"{env}-esg-scores"
    s3_folder = f"{company_secret}/internal/Company Documents"

    s3 = boto3.client("s3", aws_access_key_id = secrets.get("api_access_key_id"), aws_secret_access_key = secrets.get("api_secret_access_key") , region_name="eu-west-2")
    for file_path in downloaded_files:
        s3_key = os.path.join(s3_folder, os.path.basename(file_path))
        try:
            s3.upload_file(file_path, s3_bucket, s3_key)
            print(f"Uploaded to S3: s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"Failed to upload {file_path} to S3: {e}")


    return downloaded_files



######## GENERAL UTILITY FUNCTIONS ########

# Function to convert numpy types to native Python types
def to_native(val):
    """Convert numpy types to native Python types."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    return val

# Function to get SASB sub-industry from SIC code
def get_sasb_sub_industry(conn, sic_code):
    """
    Given a SIC code (int or string), return the corresponding SASB sub-industry.
    Returns a dict: {"sector_id": ..., "industry_id": ...} or None if not found.
    """

    if sic_code is None:
        return None

    with conn.cursor() as cur:
        query = """
        SELECT sic."SIC_CodeID", s."SectorID", i."IndustryID", s."SectorDescription", i."IndustryDescription"
        FROM "SIC_Industry" sic
        LEFT JOIN "Industry" i ON sic."IndustryID" = i."IndustryID"
        LEFT JOIN "Sector" s ON i."SectorID" = s."SectorID"
        WHERE sic."SIC_CodeID" = %s
        """
        cur.execute(query, (str(sic_code),))
        result = cur.fetchone()
        if result:
            print(f"SIC {sic_code} maps to Industry: {result[4]} in Sector: {result[3]}")
            return {"sector_id": result[1], "industry_id": result[2]}
        else:
            return None
    
        
# Function to split SIC codes, accepting string or list
def split_sic_codes(sic_input):
    """
    Accepts:
        - string: "64110, 62020"
        - single int: 64110
        - list: [64110, 62020]
    Returns a clean list of strings: ["64110", "62020"]
    """
    if sic_input is None or (isinstance(sic_input, float) and pd.isna(sic_input)):
        return []

    if isinstance(sic_input, list):
        return [str(x).strip().strip("'\"") for x in sic_input if x is not None]

    return [str(x).strip().strip("'\"") for x in re.split(r'[,\s]+', str(sic_input).strip()) if x]


# Function to assign industry columns to a DataFrame
def assign_industry_columns(df, sic_column="SICs"):
    def process_row(sic_str):
        sic_list = split_sic_codes(sic_str)
        return pd.Series(get_first_valid_sic_and_industry(sic_list))

    df[["PrimarySIC", "SASBIndustry"]] = df[sic_column].apply(process_row)
    return df 

# Function to get the first valid SIC code and its corresponding industry
def get_first_valid_sic_and_industry(conn, sic_list):
    """
    Given a list of SIC codes, return the first SIC code that maps to a valid SASB industry.
    Returns a tuple: (sic, industry_id). If no match is found, industry_id is None.
    """
    for sic in sic_list:
        result = get_sasb_sub_industry(conn, sic)
        if result and result["industry_id"]:
            return sic, result["industry_id"]
    # No valid industry found
    return (sic_list[0] if sic_list else None, None)


def get_aws_secret(secret_name, region_name="eu-west-2"):
    # Create a Secrets Manager client
    client = boto3.client("secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)

        # Decrypts secret using the associated KMS key
        if "SecretString" in response:
            secret = response["SecretString"]
            return json.loads(secret)
        else:
            # If the secret is binary
            return response["SecretBinary"]

    except ClientError as e:
        raise Exception(f"Could not retrieve secret: {e}")

######## RDS UTILITY FUNCTIONS ########

# Function to start the RDS connection
def rds_start_connection(env, secrets, port = None):

    # if env in ["dev", "uat", "prod"]:
    #     from dotenv import load_dotenv
    #     load_dotenv()

    try:
        # Load environment-specific configurations
        if env == "dev":
            remote_host = "dev-tdh-api-rds.cyeue0mvkf6t.eu-west-2.rds.amazonaws.com"
        elif env == "uat":
            remote_host = "uat-rds.c2vgh2settrb.eu-west-2.rds.amazonaws.com"
        elif env == "prod":
            remote_host = "prod-tdh-api-rds.cplufrbap42p.eu-west-2.rds.amazonaws.com"
        elif env == "dev-api":

            default_host = secrets.get("host")

            # Get DB_URL from environment
            db_url = os.getenv("DB_URL")

            if db_url:
                print("DB_URL environment variable is set.")
                host = db_url
                print(f"DB_URL environment variable is set. Using host: {host}")
            else:
                print("DB_URL environment variable is not set.")
                host = default_host
                print("DB_URL environment variable is not set. Using default host.")

            try:
                print(f"Connecting to database with username: {secrets.get('username')} and password: {secrets.get('password')} and host: {host} and port: {secrets.get('port')}")
                conn = psycopg2.connect(
                    dbname = secrets.get("dbname"),
                    user=secrets.get("username"),
                    password = secrets.get("password"),
                    host=host,
                    port=secrets.get("port")
                )
            except Exception as e:
                logging.exception("Database connection failed")
                return {"error": str(e)}, 500
            # Test the connection
            cur = conn.cursor()
            query = "SELECT * FROM \"Engine\" LIMIT 5"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("Connection test failed.")
            else:
                print("Connection test successful.")

            return conn
        elif env == "uat-api":

            default_host = secrets.get("host")

            # Get DB_URL from environment
            db_url = os.getenv("DB_URL")

            if db_url:
                host = db_url
                print(f"DB_URL environment variable is set. Using host: {host}")
            else:
                host = default_host
                print("DB_URL environment variable is not set. Using default host.")

            try:
                conn = psycopg2.connect(
                    dbname = secrets.get("dbname"),
                    user=secrets.get("username"),
                    password = secrets.get("password"),
                    host=host,
                    port=secrets.get("port")
                )
            except Exception as e:
                logging.exception("Database connection failed")
                return {"error": str(e)}, 500
            # Test the connection
            cur = conn.cursor()
            query = "SELECT * FROM \"Engine\" LIMIT 5"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("Connection test failed.")
            else:
                print("Connection test successful.")

            return conn
        
        elif env == "demo-api":

            default_host = "demo-rds.c3cs24s42ita.eu-west-2.rds.amazonaws.com"

            # Get DB_URL from environment
            db_url = os.getenv("DB_URL")

            if db_url:
                host = db_url
                print(f"DB_URL environment variable is set. Using host: {host}")
            else:
                host = default_host
                print("DB_URL environment variable is not set. Using default host.")

            try:
                conn = psycopg2.connect(
                                    dbname = secrets.get("dbname"),
                                    user=secrets.get("username"),
                                    password = secrets.get("password"),
                                    host=host,
                                    port=secrets.get("port")
                                    )
            except Exception as e:
                logging.exception("Database connection failed")
                return {"error": str(e)}, 500
            # Test the connection
            cur = conn.cursor()
            query = "SELECT * FROM \"Engine\" LIMIT 5"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("Connection test failed.")
            else:
                print("Connection test successful.")

            return conn
        
        elif env == "prod-api":

            default_host = "prod-rds.cplufrbap42p.eu-west-2.rds.amazonaws.com"

            # Get DB_URL from environment
            db_url = os.getenv("DB_URL")

            if db_url:
                host = db_url
                print(f"DB_URL environment variable is set. Using host: {host}")
            else:
                host = default_host
                print("DB_URL environment variable is not set. Using default host.")

            try:
                conn = psycopg2.connect(
                    dbname = secrets.get("dbname"),
                    user=secrets.get("username"),
                    password = secrets.get("password"),
                    host=host,
                    port=secrets.get("port")
                )
            except Exception as e:
                logging.exception("Database connection failed")
                return {"error": str(e)}, 500
            # Test the connection
            cur = conn.cursor()
            query = "SELECT * FROM \"Engine\" LIMIT 5"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("Connection test failed.")
            else:
                print("Connection test successful.")

            return conn

        else:
            raise ValueError(f"Unsupported environment: {env}")

        # Connect to the PostgreSQL server through the local port
        conn = psycopg2.connect(
            dbname="postgres",
            user="tdhmaster",
            password="!tdh4536db!",
            host=remote_host,
            port=port
        )

        # Test the connection
        cur = conn.cursor()
        query = "SELECT * FROM \"Engine\" LIMIT 5"
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("Connection test failed.")
        else:
            print("Connection test successful.")

        return conn

    except Exception as e:
        print(f"Failed to establish SSH tunnel or connect to the database: {str(e)}")
        return None

# Function to get the nearest revenue neighbour
def metrics3_return_nearest_revenue_neighbour(conn, sectorID, industryID, company_raw_revenue):
    """
    Returns the CompanyID of the nearest revenue neighbour based on SectorID and IndustryID,
    that also has a corresponding CompanyAssessmentID with AssessmentQuestionIDs.
    If no company with revenue <= provided `company_raw_revenue` has these, 
    it returns the nearest company with a MetricAssessmentID and MetricAssessmentQuestionIDs regardless of revenue.
    """
    # First query: Find the nearest neighbour with revenue <= provided value that has both a MetricAssessmentID and MetricAssessmentQuestionIDs
    query = """
    SELECT cl."CompanyID"
        FROM public."Company" cl
        JOIN public."CompanyAssessment" ma ON cl."CompanyID" = ma."CompanyID"
        JOIN public."CompanyScrape" cs ON cl."CompanyID" = cs."CompanyID"
        JOIN public."AssessmentQuestion" maq ON ma."CompanyAssessmentID" = maq."CompanyAssessmentID"
        WHERE cl."SectorID" = %s AND cl."IndustryID" = %s
        AND CAST(cs."RawRevenue" AS NUMERIC) <= %s
        AND maq."AssessmentQuestionID" IS NOT NULL  -- Ensure there's at least one AssessmentQuestionID
        ORDER BY ABS(CAST(cs."RawRevenue" AS NUMERIC) - %s)
        LIMIT 1;
    """

    with conn.cursor() as cur:
        cur.execute(query, (sectorID, industryID, to_native(company_raw_revenue), to_native(company_raw_revenue)))
        result = cur.fetchone()
    
    if result:
        return result[0]  # CompanyLeadID with both MetricAssessmentID and MetricAssessmentQuestionIDs
    else:
        # Fallback query: Find the nearest neighbour with a MetricAssessmentID and MetricAssessmentQuestionIDs (regardless of revenue)
        nearest_query = """
            SELECT cl."CompanyID"
            FROM public."Company" cl
            JOIN public."CompanyAssessment" ma ON cl."CompanyID" = ma."CompanyID"
            JOIN public."CompanyScrape" cs ON cl."CompanyID" = cs."CompanyID"
            JOIN public."AssessmentQuestion" maq ON ma."CompanyAssessmentID" = maq."CompanyAssessmentID"
            WHERE cl."SectorID" = %s AND cl."IndustryID" = %s
            AND maq."AssessmentQuestionID" IS NOT NULL  -- Ensure there's at least one AssessmentQuestionID
            ORDER BY ABS(CAST(cs."RawRevenue" AS NUMERIC) - %s)
            LIMIT 1;
        """
        with conn.cursor() as cur:
            cur.execute(nearest_query, (sectorID, industryID, to_native(company_raw_revenue)))
            nearest_result = cur.fetchone()

            if nearest_result:
                return nearest_result[0]  # CompanyLeadID with both MetricAssessmentID and MetricAssessmentQuestionIDs
            else:
                return None  # No company found with MetricAssessmentID and MetricAssessmentQuestionIDs

# Function to delete scores from the database
def rds_delete_scores(conn, company_assessment_id):
    # Delete scores from the database
    cursor = conn.cursor()

    # Step 1: Delete from MetricAssessmentQuestion
    cursor.execute("""
        DELETE FROM "AssessmentQuestion" 
        WHERE "CompanyAssessmentID" = %s
    """, (company_assessment_id,))

    # Step 2: Delete from MetricAssessmentTopic via MetricAssessmentCategory
    cursor.execute("""
        DELETE FROM "AssessmentTopic"
        WHERE "AssessmentCategoryID" IN (
            SELECT "AssessmentCategoryID"
            FROM "AssessmentCategory"
            WHERE "CompanyAssessmentID" = %s
        )
    """, (company_assessment_id,))

    # Step 3: Delete from MetricAssessmentCategory
    cursor.execute("""
        DELETE FROM "AssessmentCategory" 
        WHERE "CompanyAssessmentID" = %s
    """, (company_assessment_id,))

    conn.commit()
    cursor.close()
    print(f"Deleted all related scores and data for Company Assessment ID {company_assessment_id} from RDS.")

######## RDS REFERENCE DATA / INFORMATION FUNCTIONS ########

# Function to get metric revenue ID
def rds_get_metricrevenue_id(conn, raw_revenue):
    """
    Get MetricRevenueID from raw revenue range string or numeric value.
    
    First tries to match by description (e.g., "┬ú2mil - ┬ú5mil", "┬ú5mil-┬ú20mil").
    If that fails, tries to parse as numeric and find matching range bucket.
    """
    
    if (raw_revenue == None):
        return None
    
    # First, try to match by description (for formats like "┬ú2mil - ┬ú5mil", etc.)
    query_desc = """
        SELECT "MetricRevenueID" 
        FROM "MetricRevenue" 
        WHERE "MetricRevenueDescription" = %s
        LIMIT 1
    """
    
    with conn.cursor() as cursor:
        cursor.execute(query_desc, (raw_revenue,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
    
    # If description match failed, try to parse as numeric value
    # and find which range bucket it falls into
    try:
        # Try to extract numeric value from string (e.g., "5000000" from "┬ú5mil-┬ú20mil")
        # or use as-is if it's already numeric
        if isinstance(raw_revenue, str):
            # Remove currency symbols and "mil" text, convert to number
            import re
            # Extract numbers and handle "mil" multiplier
            cleaned = raw_revenue.replace('┬ú', '').replace(',', '').strip()
            
            # Check for "mil" or "million" multiplier
            if 'mil' in cleaned.lower() or 'million' in cleaned.lower():
                # Extract first number before "mil"
                numbers = re.findall(r'[\d.]+', cleaned)
                if numbers:
                    numeric_value = int(float(numbers[0]) * 1000000)  # Convert to actual number
                else:
                    logging.warning(f"Could not extract numeric value from revenue range: {raw_revenue}")
                    return None
            else:
                # Try to extract numbers as-is
                numbers = re.findall(r'\d+', cleaned)
                if numbers:
                    numeric_value = int(numbers[0])
                else:
                    logging.warning(f"Could not extract numeric value from revenue range: {raw_revenue}")
                    return None
        else:
            numeric_value = int(raw_revenue)
        
        # Build the SQL query to find the matching revenue range bucket
        query = """
            SELECT "MetricRevenueID" 
            FROM "MetricRevenue" 
            WHERE %s >= "StartRevenue" AND %s <= "EndRevenue"
            LIMIT 1
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (numeric_value, numeric_value))
            result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            logging.warning(f"No matching revenue range bucket found for value: {numeric_value}")
            return None
            
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse revenue range as numeric: {raw_revenue}. Error: {e}")
        return None

# Function to get number of employees ID
def rds_get_numberofemployees_id(conn, raw_employees):
    """
    Get NumberOfEmployeeID from raw employee range string or numeric value.
    
    First tries to match by description (e.g., "50 to 249", "1 to 9").
    If that fails, tries to parse as numeric and find matching range bucket.
    """
    
    # First, try to match by description (for formats like "50 to 249", "1 to 9", etc.)
    query_desc = '''
        SELECT "NumberOfEmployeeID"
        FROM "NumberOfEmployee"
        WHERE "NumberOfEmployeeDescription" = %s
        LIMIT 1
    '''
    
    with conn.cursor() as cursor:
        cursor.execute(query_desc, (raw_employees,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
    
    # If description match failed, try to parse as numeric value
    # and find which range bucket it falls into
    try:
        # Try to extract numeric value from string (e.g., "50" from "50 to 249")
        # or use as-is if it's already numeric
        if isinstance(raw_employees, str):
            # Try to extract first number from string
            import re
            numbers = re.findall(r'\d+', raw_employees)
            if numbers:
                numeric_value = int(numbers[0])
            else:
                # No numbers found, return None
                logging.warning(f"Could not extract numeric value from employee range: {raw_employees}")
                return None
        else:
            numeric_value = int(raw_employees)
        
        # Build the SQL query to find the matching employee count bucket using BETWEEN
        query = '''
            SELECT "NumberOfEmployeeID"
            FROM "NumberOfEmployee"
            WHERE %s BETWEEN "RangeStart" AND "RangeEnd"
            LIMIT 1
        '''

        # Execute the query
        with conn.cursor() as cursor:
            cursor.execute(query, (numeric_value,))
            result = cursor.fetchone()

        # Check if a result was found
        if result is None:
            logging.warning(f"No matching employee count bucket found for value: {numeric_value}")
            return None

        # Return the first result
        return result[0]
        
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse employee range as numeric: {raw_employees}. Error: {e}")
        return None

# Function to get sector ID
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
        return None

# Function to get industry ID
def rds_get_industry_id(conn, industry_desc):
    query = """
        SELECT "IndustryID" FROM "Industry" 
        WHERE "IndustryDescription" LIKE %s
        """
    with conn.cursor() as cursor:
        cursor.execute(query, (f"{industry_desc}%",))  # match industry that starts with string
        result = cursor.fetchone()  # get one matching row

    if result:
        return result[0]
    else:
        return None
    
def rds_get_question_id_for_metrics3_based_on_oldquestionnumber(conn, old_question_number):
    with conn.cursor() as cur:
        query = f"SELECT * FROM \"Question\" q LEFT JOIN \"SurveyQuestion\" sq ON q.\"QuestionID\" = sq.\"QuestionID\"  WHERE sq.\"SurveyID\" = 3333 AND \"OldQuestionNumber\" = {old_question_number}"
        cur.execute(query)
        question_id = cur.fetchall()
        question_id = question_id[0][0]
    return question_id

######## RDS COMPANY INFORMATION FUNCTIONS ########

# Function to create or get company ID
def rds_create_company_id(
    conn,
    company_name,
    company_website,
    company_revenue=None,
    company_employees=None,
    company_sector=None,
    SIC_codes=None,
    company_industry=None,
    company_secret=None,
    industry_id=None,
    partner_id=3,
    partner_name=None,
    company_house_number=None,
    employee_id=None,
    revenue_id=None
):
    """
    Fetch the CompanyID for a given company name. If no match is found, create a new company record.
    Returns CompanyID, CompanyScrapeID, warnings, and any exception.

    By default, the partner ID is set to 3, which is "The Disruption House Limited".
    """

    warnings = []
    errors = []

    if not company_secret:
        company_secret = str(uuid.uuid4())

    # Early validation - Required fields
    if not company_name:
        logging.error("No Company Name provided.")
        return {"Error": ["No Company Name provided."]}

    if not company_website:
        logging.error("No Company Website provided.")
        return {"Error": ["No Company Website provided."]}

    try:
        with conn.cursor() as cur:
            # Check if company already exists
            cur.execute(
                """
                SELECT "CompanyID" FROM public."Company"
                WHERE LOWER("CompanyName") = LOWER(%s)
                LIMIT 1;
                """,
                (company_name,)
            )
            result = cur.fetchone()
            if result:
                logging.info("Company already exists.")
                return {"CompanyID": result[0], "Warnings": warnings}

            # Process SIC codes and determine industry/sector from SIC
            sic_1 = sic_2 = sic_3 = sic_4 = None
            split_codes = []
            sic_derived_sector_id = None
            sic_derived_industry_id = None
            
            if SIC_codes:
                split_codes = split_sic_codes(SIC_codes)
                print(split_codes)
                valid_pair = get_first_valid_sic_and_industry(conn, split_codes)
                print(valid_pair)
                
                # Get the full SASB data including sector (only if company_sector not provided)
                if valid_pair and valid_pair[0] and not company_sector:
                    sasb_data = get_sasb_sub_industry(conn, valid_pair[0])
                    if sasb_data:
                        sic_derived_sector_id = sasb_data.get("sector_id")
                        sic_derived_industry_id = sasb_data.get("industry_id")
                elif valid_pair and valid_pair[0]:
                    # If company_sector was provided, still get industry from SIC
                    sic_derived_industry_id = valid_pair[1]

                # Use plain strings for SIC codes
                sic_1 = split_codes[0] if len(split_codes) > 0 else None
                sic_2 = split_codes[1] if len(split_codes) > 1 else None
                sic_3 = split_codes[2] if len(split_codes) > 2 else None
                sic_4 = split_codes[3] if len(split_codes) > 3 else None

            # EARLY VALIDATION - Convert and verify all IDs exist in database
            sector_id = None
            if company_sector:
                # User provided sector name - look it up
                sector_id = to_native(rds_get_sector_id(conn, company_sector))
                if sector_id is None:
                    errors.append(f"SectorID could not be found in the database. Are you sure the sector '{company_sector}' is correct and matches the database?")
            elif sic_derived_sector_id:
                # Use sector derived from SIC codes only if company_sector not provided
                sector_id = sic_derived_sector_id

            if company_industry:
                # User provided industry name - look it up
                industry_id = to_native(rds_get_industry_id(conn, company_industry))
            elif sic_derived_industry_id:
                # Use industry derived from SIC codes
                industry_id = sic_derived_industry_id
                warnings.append(f"Raw industry not found in the database - Industry derived from SIC code '{valid_pair[0]}'.")
            elif industry_id is not None:
                # Industry ID was provided directly
                industry_id = to_native(industry_id)
            
            if company_industry and industry_id is None:
                errors.append(f"IndustryID could not be found in the database. Are you sure the industry '{company_industry}' is correct and matches the database?")

            # Use provided IDs directly, or look them up from strings if IDs not provided
            if employee_id is None:
                employee_id = None
                if company_employees:
                    employee_id = to_native(rds_get_numberofemployees_id(conn, company_employees))
                    if employee_id is None:
                        errors.append(f"EmployeeID could not be found in the database. Are you sure the employee size '{company_employees}' is correct and matches the database?")
            else:
                # Validate that the provided employee_id exists
                employee_id = to_native(employee_id)
                cur.execute('SELECT "NumberOfEmployeeID" FROM "NumberOfEmployee" WHERE "NumberOfEmployeeID" = %s', (employee_id,))
                if not cur.fetchone():
                    errors.append(f"Invalid NumberOfEmployeeID: {employee_id}")

            if revenue_id is None:
                revenue_id = None
                if company_revenue:
                    revenue_id = to_native(rds_get_metricrevenue_id(conn, company_revenue))
                    if revenue_id is None:
                        errors.append(f"MetricRevenueID could not be found in the database. Are you sure the revenue '{company_revenue}' is correct and matches the database?")
            else:
                # Validate that the provided revenue_id exists
                revenue_id = to_native(revenue_id)
                cur.execute('SELECT "MetricRevenueID" FROM "MetricRevenue" WHERE "MetricRevenueID" = %s', (revenue_id,))
                if not cur.fetchone():
                    errors.append(f"Invalid MetricRevenueID: {revenue_id}")


            # Verify partner exists if provided
            if not partner_id and partner_name:
                cur.execute(
                    """
                    SELECT "PartnerID" FROM public."Partner"
                    WHERE LOWER("PartnerName") = LOWER(%s)
                    LIMIT 1;
                    """,
                    (partner_name,)
                )
                partner_result = cur.fetchone()
                if partner_result:
                    partner_id = partner_result[0]
                else:
                    errors.append(f"PartnerID could not be found in the database. Are you sure the partner '{partner_name}' is correct and matches the database?")


            # If there are any errors after partner validation, return early
            if errors:
                logging.error("Validation errors: %s", errors)
                return {"Error": errors, "Warnings": warnings}

            # Collect missing info warnings (non-blocking)
            if company_employees is None:
                warnings.append("Number of employees is missing.")
            if company_revenue is None:
                warnings.append("Company revenue is missing.")
            if company_sector is None and sector_id is None:
                warnings.append("Sector is missing.")
            if company_industry is None and industry_id is None:
                if SIC_codes:
                    warnings.append(f"No valid SASB industry found for SIC codes: {split_codes}")
                else:
                    warnings.append("Industry is missing.")
            if company_house_number is None:
                warnings.append("Companies House Number is missing.")

            # All validations passed - proceed with insert
            company_employees_native = to_native(company_employees) if company_employees else None
            company_revenue_native = to_native(company_revenue) if company_revenue else None

            # Insert into Company
            cur.execute(
                """
                INSERT INTO public."Company" 
                    ("CompanyName", "CompanyWebSite", "SectorID", "IndustryID", "Secret",
                     "NumberOfEmployeeID", "MetricRevenueID", "SIC_1", "SIC_2", "SIC_3", "SIC_4", 
                     "CreateDate", "CompanyHouseNumber")
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING "CompanyID";
                """,
                (company_name, company_website, sector_id, industry_id, company_secret,
                 employee_id, revenue_id, sic_1, sic_2, sic_3, sic_4, datetime.now(), company_house_number)
            )
            new_company_id = cur.fetchone()[0]

            # Insert into CompanyScrape if any raw data exists
            new_companyscrape_id = None
            if company_sector or company_industry or company_employees or company_revenue:
                cur.execute(
                    """
                    INSERT INTO public."CompanyScrape" 
                        ("CompanyID", "RawSector", "RawIndustry", "RawEmployeeSize", "RawRevenue")
                    VALUES (%s,%s,%s,%s,%s)
                    RETURNING "CompanyScrapeID";
                    """,
                    (new_company_id, company_sector, company_industry, company_employees, company_revenue)
                )
                new_companyscrape_id = cur.fetchone()[0]

            # Insert into CompanyPartner if partner_id is provided
            if partner_id:
                cur.execute(
                    """
                    INSERT INTO public."PartnerCompany" 
                        ("CompanyID", "PartnerID", "RelationshipStartDate")
                    VALUES (%s, %s, %s);
                    """,
                    (new_company_id, partner_id, datetime.now())
                )

            conn.commit()
            logging.info("New company record created.")

            return {
                "CompanyName": company_name,
                "CompanyID": new_company_id,
                "CompanyScrapeID": new_companyscrape_id,
                "Warnings": warnings
            }

    except Exception as e:
        logging.error("Error during execution: %s", e)
        conn.rollback()
        return {
            "CompanyName": company_name,
            "Error": [str(e)],
            "Warnings": warnings
        }

    
# Function to get company ID
def rds_get_company_id(conn, company_name):
    """
    Fetch the CompanyID for a given company name. If no match is found, return None.

    :param conn: A psycopg2 database connection object.
    :param company_name: The name of the company to search for.
    :return: The CompanyLeadID (existing or None).
    """
    if not company_name:
        logging.error("Invalid company name provided.")
        return None

    try:
        with conn.cursor() as cur:
            # Try to find an existing company ID
            query = """
                SELECT "CompanyID" FROM public."Company"
                WHERE LOWER("CompanyName") = LOWER(%s)
                LIMIT 1;
            """
            cur.execute(query, (company_name,))
            result = cur.fetchone()

            if result:
                logging.info("Successfully retrieved the company ID.")
                return result[0]
            else:
                logging.info("No matching company ID found.")
                return None

    except Exception as e:
        logging.error(f"Error during execution: {e}")
        return None

# Function to get metric company secret
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

# Function to get metric category scores
def rds_get_metric_category_scores(conn, company_assessment_id):
    #Fetch results
    fetch_query = """SELECT ac."AssessmentCategoryID", cat."CategoryDescription", ac."SummaryScore", ac."CompanyAssessmentID" FROM public."AssessmentCategory" ac
        LEFT JOIN "Category" cat ON ac."CategoryID" = cat."CategoryID" 
        WHERE "CompanyAssessmentID" = %(company_assessment_id)s"""
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"company_assessment_id": company_assessment_id})
        print("Successfully fetched updated category scores.")
    except Exception as e:
        print(f"Error fetching updated category scores: {e}")
        results = None

    return(results)

# Function to get metric category scores from companyleadid
def rds_get_metric_category_scores_from_companyid(conn, company_id):
    #Fetch results
    fetch_query = """SELECT ac."AssessmentCategoryID", cat."CategoryDescription", ac."SummaryScore", ac."CompanyAssessmentID" FROM public."AssessmentCategory" ac
    LEFT JOIN "CompanyAssessment" ma ON ma."CompanyAssessmentID" = ac."CompanyAssessmentID"
    LEFT JOIN "Category" cat ON ac."CategoryID" = cat."CategoryID" 
    WHERE "CompanyID" = %(company_id)s"""
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"company_id": company_id})
        print("Successfully fetched category scores.")
    except Exception as e:
        print(f"Error fetching category scores: {e}")
        results = None

    return(results)

# Function to get metric topic scores from companyleadid
def rds_get_metric_topic_scores_from_companyid(conn, company_id):
    #Fetch results
    fetch_query = """SELECT "AssessmentTopicID",top."TopicDescription", mat."Score", cat."CategoryDescription"  FROM "AssessmentTopic" mat
    LEFT JOIN "Topic" top ON mat."TopicID" = top."TopicID"
    LEFT JOIN "AssessmentCategory" mac ON mac."AssessmentCategoryID" = mat."AssessmentCategoryID"
    LEFT JOIN "Category" cat ON cat."CategoryID" = mac."CategoryID"
    WHERE mat."AssessmentCategoryID" IN (
        SELECT ac."AssessmentCategoryID" FROM public."AssessmentCategory" ac
        LEFT JOIN "CompanyAssessment" ma ON ma."CompanyAssessmentID" = ac."CompanyAssessmentID"
        LEFT JOIN "Category" cat ON ac."CategoryID" = cat."CategoryID" 
        WHERE "CompanyID" = %(company_id)s)"""
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"company_id": company_id})
        print("Successfully fetched topic scores.")
    except Exception as e:
        print(f"Error fetching topic scores: {e}")
        results = None

    return(results)

# Function to get metric topic scores
def rds_get_metric_topic_scores(conn, company_assessment_id):
    # Fetch results
    fetch_query = """
        SELECT mat."AssessmentTopicID",
        top."TopicDescription",
        mat."Score",
        mat."AssessmentCategoryID"
        FROM public."AssessmentTopic" mat
        LEFT JOIN public."AssessmentCategory" mac ON mat."AssessmentCategoryID" = mac."AssessmentCategoryID"
        LEFT JOIN public."Topic" top ON mat."TopicID" = top."TopicID"
        WHERE mac."CompanyAssessmentID" = %(company_assessment_id)s
    """
    try:
        results = pd.read_sql_query(fetch_query, conn, params={"company_assessment_id": company_assessment_id})
        print("Successfully fetched topic scores.")
    except Exception as e:
        print(f"Error fetching topic scores: {e}")
        results = None

    return results

# Function to get company information based on CompanyLeadID or CompaniesHouseNumber
def rds_get_company_information(conn, company_id=None, company_house_number=None, all_columns=False):
    """
    Fetch company information based on CompanyLeadID or CompanyHouseNunmber.
    
    :param conn: A psycopg2 database connection object.
    :param companylead_id: The CompanyLeadID to search for.
    :param companies_house_number: The CompanyHouseNumber to search for.
    :param all_columns: Boolean to fetch all columns or a simplified version with joins.
    :return: DataFrame containing company information.
    """

    if all_columns:
        query = """
            SELECT * FROM public."Company" cl
            LEFT JOIN "CompanyScrape" cs ON cl."CompanyID" = cs."CompanyID"
            WHERE cl."CompanyID" = %s OR cl."CompanyHouseNumber" = %s
        """
    else:
        query = """
            SELECT 
                cl."CompanyName", 
                cl."CompanyWebSite", 
                cs."RawSector", 
                cs."RawIndustry", 
                e."NumberOfEmployeeDescription" AS "NumberOfEmployees", 
                r."MetricRevenueDescription" AS "RevenueRange",
                cl."SectorID"
            FROM public."Company" cl
            LEFT JOIN "Industry" i ON cl."IndustryID" = i."IndustryID"
            LEFT JOIN "Sector" s ON cl."SectorID" = s."SectorID"
            LEFT JOIN "MetricRevenue" r ON cl."MetricRevenueID" = r."MetricRevenueID"
            LEFT JOIN "NumberOfEmployee" e ON cl."NumberOfEmployeeID" = e."NumberOfEmployeeID"
            LEFT JOIN "CompanyScrape" cs ON cl."CompanyID" = cs."CompanyID"
            WHERE cl."CompanyID" = %s OR cl."CompanyHouseNumber" = %s
        """

    with conn.cursor() as cursor:
        cursor.execute(query, (company_id, company_house_number))
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

    if rows:
        return pd.DataFrame(rows, columns=column_names)
    else:
        return pd.DataFrame()  # Return empty DataFrame instead of None

# Funtion to get partner ID
def rds_get_partner_id(conn, partner_name):
    with conn.cursor() as cursor:
        query = """
            SELECT "PartnerID" FROM "Partner" WHERE "PartnerName" = %s
        """
        cursor.execute(query, (partner_name,))
        result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return "No partner found"

# Function to update company information
def rds_update_company_information(conn, company_id, company_name=None, metricrevenue_id=None, numberofemployees_id=None, sector_id=None, industry_id=None):
    # this function updates the company information in the database dependent on which fields are provided
    # Create a dictionary to hold the fields to be updated
    update_fields = {}
    if company_name:
        update_fields['CompanyName'] = company_name
    if metricrevenue_id:
        update_fields['MetricRevenueID'] = metricrevenue_id
    if numberofemployees_id:
        update_fields['NumberOfEmployeeID'] = numberofemployees_id
    if sector_id:
        update_fields['SectorID'] = sector_id
    if industry_id:
        update_fields['IndustryID'] = industry_id
    if not update_fields:
        print("No fields to update.")
        return
    # Create the SET clause of the SQL query
    set_clause = ", ".join([f'"{key}" = %s' for key in update_fields.keys()])
    # Create the values tuple
    values = tuple(update_fields.values())
    # Create the SQL query
    query = f"""
        UPDATE public."Company"
        SET {set_clause}
        WHERE "CompanyID" = %s
    """
    # Add the companylead_id to the values tuple
    values += (company_id,)
    # Execute the query
    with conn.cursor() as cursor:
        cursor.execute(query, values)
        conn.commit()
        print(f"CompanyLeadID {company_id} updated successfully.")

    # I want to return the company lead id and new values
    updated_values = {**update_fields, 'CompanyID': company_id}
    return updated_values

######## RDS SCORING FUNCTIONS ########

# Function to create or get metric assessment ID
def rds_create_or_get_companyassessment_id(conn, company_id, product_id, survey_id = None):
    """
    Fetch the CompanyAssessmentID for a given company ID and product ID. 
    If no match is found on the same day, create a new company assessment record.
    
    :param conn: A psycopg2 database connection object.
    :param company_id: The company ID to search for or insert.
    :param product_id: The product ID.
    :return: The MetricAssessmentID (existing or newly created).
    """
    if not company_id or not product_id:
        logging.error("Invalid company ID or product ID provided.")
        return None
    
    try:
        with conn.cursor() as cur:
            # Try to find an existing CompanyAssessmentID for today
            query = """
                SELECT "CompanyAssessmentID" FROM public."CompanyAssessment"
                WHERE "CompanyID" = %s AND "ProductID" = %s 
                LIMIT 1;
            """
            cur.execute(query, (company_id, product_id))
            result = cur.fetchone()
            
            if result:
                logging.info("Successfully retrieved the existing CompanyAssessmentID.")
                return result[0]
            
            # If no match, insert a new record with full timestamp
            insert_query = """
                INSERT INTO public."CompanyAssessment" ("CompanyID", "ProductID", "CreateDate", "SurveyID")
                VALUES (%s, %s, NOW(), %s) RETURNING "CompanyAssessmentID";
            """
            cur.execute(insert_query, (company_id, product_id, survey_id))
            new_assessment_id = cur.fetchone()[0]
            conn.commit()
            
            logging.info("New company assessment record created.")
            return new_assessment_id
    
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        conn.rollback()
        return None

# Function to insert manual scores into the MetricAssessmentQuestion table
def rds_insert_manual_scores_into_assessmentquestion(conn, scores):
    for index, rows in scores.iloc[:120].iterrows():  # Process only first 120 rows
        companyassessment_id = int(rows["CompanyAssessmentID"])
        old_question_number = rows['OldQuestionNumber']

        # Handle NaN manual scores
        if math.isnan(rows['manual_score']):
            manual_score = 0
        else:
            manual_score = int(rows['manual_score'])

        answer_text = rows["Website_Reference"]

        question_id = int(rds_get_question_id_for_metrics3_based_on_oldquestionnumber(conn, old_question_number))
        
        with conn.cursor() as cur:
            query = """
                INSERT INTO "AssessmentQuestion" ("QuestionID", "Score", "CompanyAssessmentID", "AnswerText")
                VALUES (%s, %s, %s, %s)
            """
            cur.execute(query, (question_id, manual_score, companyassessment_id, answer_text))
            conn.commit()
    
    return "Scores inserted successfully."

# Function to get Metric 3 scores
def rds_get_metrics_3_scores(conn, company_id=None, companyassessment_id=None):
    """
    Fetch Metric 3 scores based on either companylead_id or metricassessment_id.
    """

    if company_id is None and companyassessment_id is None:
        raise ValueError("Either companylead_id or metricassessment_id must be provided.")

    base_query = """
        SELECT cl."CompanyID", aq."AssessmentQuestionID", q."OldQuestionNumber", 
               p."PromptQuestion", aq."AnswerText", aq."Score", aq."CompanyAssessmentID"
        FROM "AssessmentQuestion" aq
        LEFT JOIN "Question" q ON aq."QuestionID" = q."QuestionID"
        LEFT JOIN "Prompt" p ON q."PromptID" = p."PromptID"
        LEFT JOIN "CompanyAssessment" ma ON ma."CompanyAssessmentID" = aq."CompanyAssessmentID"
        LEFT JOIN "Company" cl ON cl."CompanyID" = ma."CompanyID"
        WHERE {} = %s
        ORDER BY q."QuestionID" ASC
    """

    param_field = 'cl."CompanyID"' if companyassessment_id is None else 'aq."CompanyAssessmentID"'
    param_value = company_id if companyassessment_id is None else companyassessment_id
    query = base_query.format(param_field)

    try:
        with conn.cursor() as cur:
            cur.execute(query, (param_value,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        logging.error("Error getting Metrics 3 scores: %s", e)
        return []

# Function to get metric assessment ID
def rds_get_evaluation_type_description(conn, companyassessment_id):
    # Get the EvaluationTypeID from the database
    cursor = conn.cursor()
    cursor.execute("SELECT \"EvaluationTypeDescription\" FROM \"CompanyAssessment\" LEFT JOIN \"EvaluationType\" ON \"CompanyAssessment\".\"EvaluationTypeID\" = \"EvaluationType\".\"EvaluationTypeID\" WHERE \"CompanyAssessment\".\"CompanyAssessmentID\" = %s", (companyassessment_id,))
    result = cursor.fetchone()
    cursor.close()
    if result:
        return result[0]
    else:
        conn.rollback()
        raise ValueError(f"Evaluation type for  '{companyassessment_id}' not found in the database.")
    
# Function to insert topic scores
def rds_insert_category_scores(conn):
    
    with conn.cursor() as cur:
        query = """
                insert into public."AssessmentCategory"
                                ("CompanyAssessmentID","CategoryID","SummaryScore")
                                select distinct "CompanyAssessmentID",a."CategoryID",
                                sum((("MetricWeightedScore"/100)*(ac."Weight"/100))::numeric(7,2))as "CategoryWeightScore"
                                from public.view_assessment_metricscore  a
                                join public."ApplicabilityCategory" ac on ac."CategoryID" = a."CategoryID" and ac."TopicID" = a."TopicID" and ac."SurveyID" = a."SurveyID"
                                where a."SurveyID" =3333
                                and "CompanyAssessmentID" not in (select distinct ca."CompanyAssessmentID"
                                                from public."AssessmentCategory" ma
                                                        join "CompanyAssessment" ca on ca."CompanyAssessmentID" = ma."CompanyAssessmentID"
                                                        and "SurveyID" =3333)
                                group by "CompanyAssessmentID",a."CategoryID";
        """
        
        cur.execute(query)
        conn.commit()
        
    return("Category scores inserted successfully.")

# Function to insert category scores
def rds_insert_topic_scores(conn):

    with conn.cursor() as cur:
        query = """insert into public."AssessmentTopic"
                    ("AssessmentCategoryID","TopicID","Score")
                    select distinct ac."AssessmentCategoryID",a."TopicID",sum(("MetricWeightedScore"/100)::numeric(7,2))as "TopicWeightScore"
                    from public.view_assessment_metricscore a
                    join public."AssessmentCategory" ac on ac."CategoryID" = a."CategoryID"
                    and ac."CompanyAssessmentID" = a."CompanyAssessmentID"
                    where a."SurveyID" =3333
                    and "AssessmentCategoryID" not in (select distinct at."AssessmentCategoryID"
                                    from public."AssessmentTopic" at
                                            join "AssessmentCategory" ac on ac."AssessmentCategoryID" = at."AssessmentCategoryID"
                                            join "CompanyAssessment" ca on ca."CompanyAssessmentID" = ac."CompanyAssessmentID"
                                            and "SurveyID" =3333)
                    group by ac."AssessmentCategoryID",a."TopicID";
         """
        cur.execute(query)
        conn.commit()

    return("Topic scores inserted successfully.")

# Function to insert extrapolated metric assessment question
def rds_insert_extrapolated_metricassessmentquestion_scores(conn, source_metric_scores, target_companyassessment_id):
        
        target_metric_scores = source_metric_scores.copy()
        target_metric_scores['CompanyAssessmentID'] = target_companyassessment_id
        for index, rows in target_metric_scores.iterrows():
            metricassessment_id = int(rows["CompanyAssessmentID"])
            old_question_number = rows['OldQuestionNumber']
            extraploated_score = int(rows['Score'])

            question_id = int(esg.rds_get_question_id_for_metrics3_based_on_oldquestionnumber(conn, old_question_number))
            
            with conn.cursor() as cur:
                query = """
                    INSERT INTO "AssessmentQuestion" ("QuestionID", "Score", "CompanyAssessmentID")
                    VALUES (%s, %s, %s)
                """
                cur.execute(query, (question_id, extraploated_score, metricassessment_id))

                # Update the CompanyAssessment table with the correct EvaluationTypeID
                update_query = """
                    UPDATE public."CompanyAssessment"
                    SET "EvaluationTypeID" = 3
                    WHERE "CompanyAssessmentID" = %s;
                    """
                
                cur.execute(update_query, (target_companyassessment_id,))

                conn.commit()

        return "Extrapolated question scores inserted successfully."

######## WEBFORM DROPDOWN AND INTERACTION FUNCTIONS ########

# Function to get all metrics 3 assessments
def rds_get_all_metrics3_assessments(conn, company_details=True, json=False):
    """
    Fetch all Metrics 3 companies from the CompanyAssessment table. If company_details is True, return only company details like name, industry, sector, revenue and employee count.
    param conn: A psycopg2 database connection object.
    param only_names: Boolean flag to return only company names.
    return: DataFrame containing company names and IDs.
    """
    
    query = """ SELECT * FROM "CompanyAssessment" ma 
        LEFT JOIN "Company" cl ON ma."CompanyID" = cl."CompanyID"
        LEFT JOIN "Industry" i ON cl."IndustryID" = i."IndustryID"
        LEFT JOIN "Sector" s ON cl."SectorID" = s."SectorID"
        LEFT JOIN "MetricRevenue" r ON cl."MetricRevenueID" = r."MetricRevenueID"
        LEFT JOIN "NumberOfEmployee" e ON cl."NumberOfEmployeeID" = e."NumberOfEmployeeID"
        WHERE ma."SurveyID" = 3333
    """

    try:
        results = pd.read_sql_query(query, conn)
        print("Successfully fetched all companies.")

        if company_details:
            results = results[['CompanyName', "IndustryDescription", "SectorDescription", "MetricRevenueDescription", "NumberOfEmployeeDescription"]]
            

    
    except Exception as e:
        print(f"Error fetching all companies: {e}")
        

    if json: 
        results = results.to_json(orient='records')
    return results

def email_client_validation_request(conn, client_email, company_id):
    # secret = get_aws_secret("Dev-Secret-Environment")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"Trying to send email to {client_email}")


    logger.info("Request acknowledged. Starting process...")

    # Print the secret values
    # smtp_username = secret["smtp_user"]
    smtp_username = "AKIARWTAIRLVRZCIKVHD"
    # smtp_password = secret["smtp_password"]
    smtp_password = "BLZwppG2ju82LwoxXQ2Jquiad0WCUUiNGRoyLK0xtQ/b"
    # sender_email = secret["ses_sender"]
    sender_email = "dev-aws@TheDisruptionHouse.com"
    # smtp_host = secret["smtp_host"]
    smtp_host = "email-smtp.eu-west-2.amazonaws.com"
    # smtp_port = secret["smtp_port"]
    smtp_port = 587

    # SES SMTP endpoint (region-specific)
    smtp_server = smtp_host
    smtp_port = smtp_port 

    # Email details
    sender_email = sender_email
    receiver_email = client_email
    subject = "ESG Essentials - Client Validation Request"
    url = "https://thedisruptionhouse.com"
    base_url = "https://metrics.esgessentials.net/"
    client_url = f"{base_url}?company_id={company_id}"

    logger.info(f"Sending email to: {receiver_email}")
    logger.info(f"Client URL: {client_url}")


    # Create the email
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    # Get company information
    company_information = rds_get_company_information(conn, company_id=company_id)
    company_name = company_information['CompanyName'].iloc[0]
    company_revenue_range = company_information['RevenueRange'].iloc[0]
    company_employee_range = company_information['NumberOfEmployees'].iloc[0]
    company_sector = company_information['RawSector'].iloc[0]
    company_industry = company_information['RawIndustry'].iloc[0]

    logger.info(f"Company Name: {company_name}")
    logger.info(f"Company Revenue Range: {company_revenue_range}")
    logger.info(f"Company Employee Range: {company_employee_range}")
    logger.info(f"Company Sector: {company_sector}")
    logger.info(f"Company Industry: {company_industry}")


    # Attach the logo image (make sure it's in your working directory)
    logo_url = "https://dev-scorecard-assets.s3.eu-west-2.amazonaws.com/images/horizontal_logo.jpg" 
    html = f"""\
    <html>
    <body style="margin:0; padding:0; background-color:#f4f4f4;">
        <table width="100%" bgcolor="#f4f4f4" cellpadding="0" cellspacing="0" border="0">
        <tr>
            <td>
            <table align="center" width="1250" cellpadding="10" cellspacing="0" border="0" style="background-color:#ffffff; margin:40px auto; padding:20px; border-radius:8px;">
                <tr>
                <td align="center" style="padding: 20px 0;">
                    <img src="{logo_url}" alt="The Disruption House Logo" width="500" style="display:block; margin-bottom: 20px;" />
                </td>
                </tr>
                <tr>
                <td align="center" style="font-family:Arial, sans-serif; padding:20px;">
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    Many companies like yours are being asked about sustainability. ItΓÇÖs important to your customers, staff and stakeholders. The average company now receives more than six supplier questionnaires in a year, some in a single month. The questionnaires are time consuming and often confusing.
                    </p>

                    <p style="color:#555555; font-size:16px; margin:20px 0;"> <b>
                    By taking 2 minutes to confirm some basic company details we can provide a free automated service to answer most, if not all, of your supplier questionnaires.
                    </b> </p> 

                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    Please confirm that the below company profile is accurate or select modify to correct the profile or provide additional detail.
                    </p>

                    <div style="border:1px solid #cccccc; border-radius:8px; padding:20px; background-color:#fafafa; width:80%; margin:20px auto; font-family:Arial, sans-serif; font-size:16px; color:#555555; text-align:left; line-height:1.6;">
                        <p><strong>Company Name:</strong> {company_name} </p>
                        <p><strong>Revenue Range:</strong> {company_revenue_range} </p>
                        <p><strong>Employees:</strong> {company_employee_range} </p>
                        <p><strong>Sector:</strong> {company_sector} </p>
                        <p><strong>Industry:</strong> {company_industry} </p>
                        <p><strong>Emission Intensity*:</strong> Low</p>
                    </div>

                    <!-- Button -->
                    <div style="text-align:center;">
                    <a href="{url}" 
                    style="display:inline-block; margin:0 10px; padding:10px 28px; font-size:15px; color:#ffffff; background-color:#008087; text-decoration:none; border-radius:5px; font-weight:bold; text-align:center; line-height:40px;">
                    Confirm
                    </a>
                    
                    <a href="{client_url}" 
                    style="display:inline-block; margin:0 10px; padding:10px 28px; font-size:15px; color:#ffffff; background-color:#008087; text-decoration:none; border-radius:5px; font-weight:bold; text-align:center; line-height:40px;">
                    Modify
                    </a>
                    </div>

                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    <b> Emission Intensity Profiles for Professional Services </b>
                    </p>

                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    The Disruption House has created sustainability profiles of the 34,000+ UK businesses with turnover between ┬ú5-100M. The profiles are based on our proprietary algorithm and available public and private data including third party sources to provide the most accurate emissions profiles available.
                    </p>

                    <div style="text-align:center; margin:40px 0;">
                    
                    <!-- Container for all boxes -->
                    <div style="display:inline-block; margin:0 15px; border:1px solid #cccccc; border-radius:8px; padding:20px; width:220px; vertical-align:top;">
                        <p style="font-weight:bold; font-size:18px; margin-bottom:10px;">Low Intensity</p>
                        <p style="color:#555555; font-size:16px; margin:10px 0;">
                        Remote staff, no business travel<br><br>
                        <strong>123 tCO2e</strong>
                        </p>
                    </div>

                    <div style="display:inline-block; margin:0 15px; border:1px solid #cccccc; border-radius:8px; padding:20px; width:220px; vertical-align:top;">
                        <p style="font-weight:bold; font-size:18px; margin-bottom:10px;">Medium Intensity</p>
                        <p style="color:#555555; font-size:16px; margin:10px 0;">
                        One office, some business travel<br><br>
                        <strong>234 tCO2e</strong>
                        </p>
                    </div>

                    <div style="display:inline-block; margin:0 15px; border:1px solid #cccccc; border-radius:8px; padding:20px; width:220px; vertical-align:top;">
                        <p style="font-weight:bold; font-size:18px; margin-bottom:10px;">High Intensity</p>
                        <p style="color:#555555; font-size:16px; margin:10px 0;">
                        Multiple offices, frequent business travel<br><br>
                        <strong>345 tCO2e</strong>
                        </p>
                    </div>

                    </div>


                </div>
                </td>
                </tr>
            </table>
            </td>
        </tr>
        </table>
    </body>
    </html>
    """

    # Attach both plain text and HTML versions
    message.attach(MIMEText(html, "html"))

    logger.info("HTML email created.")
    logger.info("Sending email...")
    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        logger.info("Email sent via SMTP!")
        return {"client_email": client_email, "email_sent": True}
    except Exception as e:
        logger.info("Failed to send email:", e)
        return {"client_email": client_email, "email_sent": False}

######## USER LOOKUP FUNCTIONS ########

def rds_get_user_info(conn, user_id):
    """
    Retrieve user information from the User table.
    
    Args:
        conn: psycopg2 database connection object
        user_id (int): UserID to look up
        
    Returns:
        dict: User information with keys:
            - UserID (int)
            - UserName (str)
            - FirstName (str)
            - Surname (str)
            - EmailAddress (str)
            - IsActive (bool)
        None: If user not found or database error
        
    Example:
        >>> user = rds_get_user_info(conn, 123)
        >>> print(user['EmailAddress'])
        'john.doe@example.com'
    """
    logger = logging.getLogger()
    
    try:
        with conn.cursor() as cursor:
            query = '''
                SELECT "UserID", "UserName", "FirstName", "Surname", 
                       "EmailAddress", "IsActive"
                FROM "User"
                WHERE "UserID" = %s
            '''
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()
            
            if row:
                user_info = {
                    'UserID': row[0],
                    'UserName': row[1],
                    'FirstName': row[2],
                    'Surname': row[3],
                    'EmailAddress': row[4],
                    'IsActive': row[5]
                }
                logger.info(f"Retrieved user info for UserID {user_id}: {user_info['UserName']}")
                return user_info
            else:
                logger.warning(f"User not found: UserID {user_id}")
                return None
                
    except Exception as e:
        logger.error(f"Error retrieving user info for UserID {user_id}: {e}")
        return None

######## EMAIL NOTIFICATION FUNCTIONS ########

def email_assessment_notification(conn, notification_email, company_id, status, 
                                   company_name=None, company_website=None, 
                                   lead_run_id=None, first_name=None, 
                                   error_message=None, secrets=None):
    """
    Send email notification for assessment workflow events.
    
    Args:
        conn: Database connection
        notification_email (str): Recipient email address
        company_id (int): Company ID
        status (str): Notification type - "triggered", "completed", 
                     "scraping_error", "scoring_error"
        company_name (str, optional): Company name
        company_website (str, optional): Company website
        lead_run_id (int, optional): LeadRun ID for tracking
        first_name (str, optional): User's first name for personalization
        error_message (str, optional): Error details for failure emails
        secrets (dict, optional): AWS secrets for SMTP credentials
        
    Returns:
        dict: {'email_sent': bool, 'recipient': str, 'status': str}
        
    Example:
        >>> result = email_assessment_notification(
        ...     conn=conn,
        ...     notification_email="user@example.com",
        ...     company_id=123,
        ...     status="triggered",
        ...     company_name="Example Corp",
        ...     first_name="John"
        ... )
        >>> print(result['email_sent'])
        True
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Validate inputs
    if not notification_email or not company_id or not status:
        logger.error("Missing required parameters for email notification")
        return {'email_sent': False, 'recipient': notification_email, 'status': status}
    
    # Validate status
    valid_statuses = ['triggered', 'completed', 'scraping_error', 'scoring_error']
    if status not in valid_statuses:
        logger.error(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return {'email_sent': False, 'recipient': notification_email, 'status': status}
    
    logger.info(f"Sending {status} email to {notification_email} for company {company_id}")
    
    try:
        # Get company information if not provided
        if not company_name or not company_website:
            try:
                company_info = rds_get_company_information(conn, company_id=company_id)
                if not company_info.empty:
                    if not company_name and 'CompanyName' in company_info.columns:
                        company_name = company_info['CompanyName'].iloc[0]
                    if not company_website and 'CompanyWebSite' in company_info.columns:
                        company_website = company_info['CompanyWebSite'].iloc[0]
            except Exception as company_error:
                logger.warning(f"Could not retrieve company information: {company_error}")
        
        # Get SMTP credentials from secrets
        if secrets:
            smtp_username = secrets.get("smtp_user", "AKIARWTAIRLVRZCIKVHD")
            smtp_password = secrets.get("smtp_password", "BLZwppG2ju82LwoxXQ2Jquiad0WCUUiNGRoyLK0xtQ/b")
            sender_email = secrets.get("ses_sender", "dev-aws@TheDisruptionHouse.com")
            smtp_host = secrets.get("smtp_host", "email-smtp.eu-west-2.amazonaws.com")
            smtp_port = int(secrets.get("smtp_port", 587))
        else:
            # Fallback to hardcoded values (dev only)
            smtp_username = "AKIARWTAIRLVRZCIKVHD"
            smtp_password = "BLZwppG2ju82LwoxXQ2Jquiad0WCUUiNGRoyLK0xtQ/b"
            sender_email = "dev-aws@TheDisruptionHouse.com"
            smtp_host = "email-smtp.eu-west-2.amazonaws.com"
            smtp_port = 587
        
        # Construct email based on status
        subject, html_body = _construct_assessment_email(
            status=status,
            company_name=company_name,
            company_website=company_website,
            company_id=company_id,
            lead_run_id=lead_run_id,
            first_name=first_name,
            error_message=error_message
        )
        
        # Create email message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = notification_email
        message.attach(MIMEText(html_body, "html"))
        
        logger.info(f"Sending email with subject: {subject}")
        
        # Send email with timeout
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(sender_email, notification_email, message.as_string())
            
            logger.info(f"Email sent successfully to {notification_email}")
            return {'email_sent': True, 'recipient': notification_email, 'status': status}
            
        except Exception as smtp_error:
            logger.error(f"SMTP error sending email: {smtp_error}")
            return {'email_sent': False, 'recipient': notification_email, 'status': status}
    
    except Exception as e:
        logger.error(f"Error sending assessment notification email: {e}")
        return {'email_sent': False, 'recipient': notification_email, 'status': status}


def _construct_assessment_email(status, company_name, company_website, company_id, 
                                 lead_run_id, first_name, error_message):
    """
    Construct email subject and HTML body based on assessment status.
    
    Args:
        status (str): Email type - triggered, completed, scraping_error, scoring_error
        company_name (str): Company name
        company_website (str): Company website URL
        company_id (int): Company ID
        lead_run_id (int): LeadRun ID for tracking
        first_name (str): User's first name for personalization
        error_message (str): Error details for failure emails
    
    Returns:
        tuple: (subject, html_body)
    """
    from datetime import datetime
    
    # Personalized greeting
    greeting = f"Hi {first_name}," if first_name else "Hello,"
    
    # Logo and base URL
    logo_url = "https://dev-scorecard-assets.s3.eu-west-2.amazonaws.com/images/horizontal_logo.jpg"
    base_url = "https://metrics.esgessentials.net/"
    
    # Common HTML header
    html_header = f"""
    <html>
    <body style="margin:0; padding:0; background-color:#f4f4f4;">
        <table width="100%" bgcolor="#f4f4f4" cellpadding="0" cellspacing="0" border="0">
        <tr>
            <td>
            <table align="center" width="1250" cellpadding="10" cellspacing="0" border="0" 
                   style="background-color:#ffffff; margin:40px auto; padding:20px; border-radius:8px;">
                <tr>
                <td align="center" style="padding: 20px 0;">
                    <img src="{logo_url}" alt="The Disruption House Logo" width="500" 
                         style="display:block; margin-bottom: 20px;" />
                </td>
                </tr>
                <tr>
                <td align="center" style="font-family:Arial, sans-serif; padding:20px;">
                    <p style="color:#333333; font-size:18px; margin:20px 0; text-align:left;">
                    {greeting}
                    </p>
    """
    
    html_footer = """
                    <p style="color:#555555; font-size:14px; margin:40px 0 20px 0; text-align:left;">
                    Best regards,<br>
                    The Disruption House Team
                    </p>
                </td>
                </tr>
            </table>
            </td>
        </tr>
        </table>
    </body>
    </html>
    """
    
    # Email content based on status
    if status == "triggered":
        subject = f"ESG Assessment Triggered - {company_name or 'Your Company'}"
        body_content = f"""
                    <h2 style="color:#008087; margin-top:0;">Assessment Started</h2>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    Your ESG assessment has been successfully queued and is now processing.
                    </p>
                    
                    <div style="border:1px solid #cccccc; border-radius:8px; padding:20px; 
                                background-color:#fafafa; margin:20px 0; text-align:left;">
                        <p><strong>Company:</strong> {company_name or 'N/A'}</p>
                        <p><strong>Website:</strong> {company_website or 'N/A'}</p>
                        <p><strong>Company ID:</strong> {company_id}</p>
                        {f'<p><strong>Job ID:</strong> {lead_run_id}</p>' if lead_run_id else ''}
                        <p><strong>Started:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    </div>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    The assessment typically takes <strong>10-15 minutes</strong> to complete.
                    We'll send you another email when it's finished.
                    </p>
        """
    
    elif status == "completed":
        subject = f"ESG Assessment Complete - {company_name or 'Your Company'}"
        results_url = f"{base_url}?company_id={company_id}"
        body_content = f"""
                    <h2 style="color:#008087; margin-top:0;">✓ Assessment Complete</h2>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    Great news! Your ESG assessment has been completed successfully.
                    </p>
                    
                    <div style="border:1px solid #cccccc; border-radius:8px; padding:20px; 
                                background-color:#fafafa; margin:20px 0; text-align:left;">
                        <p><strong>Company:</strong> {company_name or 'N/A'}</p>
                        <p><strong>Company ID:</strong> {company_id}</p>
                        {f'<p><strong>Job ID:</strong> {lead_run_id}</p>' if lead_run_id else ''}
                        <p><strong>Completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    </div>
                    
                    <div style="text-align:center; margin:30px 0;">
                        <a href="{results_url}" 
                           style="display:inline-block; padding:15px 40px; font-size:16px; 
                                  color:#ffffff; background-color:#008087; text-decoration:none; 
                                  border-radius:5px; font-weight:bold;">
                        View Assessment Results
                        </a>
                    </div>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    Click the button above to view your detailed assessment results and insights.
                    </p>
        """
    
    elif status == "scraping_error":
        subject = f"ESG Assessment Failed - Scraping Error - {company_name or 'Your Company'}"
        body_content = f"""
                    <h2 style="color:#d9534f; margin-top:0;">⚠ Assessment Error</h2>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    We encountered an error while scraping the website for your ESG assessment.
                    </p>
                    
                    <div style="border:1px solid #d9534f; border-radius:8px; padding:20px; 
                                background-color:#fff5f5; margin:20px 0; text-align:left;">
                        <p><strong>Company:</strong> {company_name or 'N/A'}</p>
                        <p><strong>Website:</strong> {company_website or 'N/A'}</p>
                        <p><strong>Company ID:</strong> {company_id}</p>
                        {f'<p><strong>Job ID:</strong> {lead_run_id}</p>' if lead_run_id else ''}
                        {f'<p><strong>Error:</strong> {error_message}</p>' if error_message else ''}
                    </div>
                    
                    <h3 style="color:#333333; font-size:16px;">Possible Causes:</h3>
                    <ul style="color:#555555; font-size:16px; text-align:left; line-height:1.8;">
                        <li>Website may be blocking automated access</li>
                        <li>Website may be temporarily unavailable</li>
                        <li>Connection timeout or network issues</li>
                        <li>SSL certificate problems</li>
                    </ul>
                    
                    <h3 style="color:#333333; font-size:16px;">Next Steps:</h3>
                    <ul style="color:#555555; font-size:16px; text-align:left; line-height:1.8;">
                        <li>Verify the website URL is correct and accessible</li>
                        <li>Try triggering the assessment again</li>
                        <li>Contact support if the issue persists</li>
                    </ul>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    <strong>Need help?</strong> Contact us at 
                    <a href="mailto:dev-aws@TheDisruptionHouse.com" style="color:#008087;">
                    dev-aws@TheDisruptionHouse.com</a>
                    </p>
        """
    
    elif status == "scoring_error":
        subject = f"ESG Assessment Failed - Scoring Error - {company_name or 'Your Company'}"
        body_content = f"""
                    <h2 style="color:#d9534f; margin-top:0;">⚠ Assessment Error</h2>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    We encountered an error while generating scores for your ESG assessment.
                    </p>
                    
                    <div style="border:1px solid #d9534f; border-radius:8px; padding:20px; 
                                background-color:#fff5f5; margin:20px 0; text-align:left;">
                        <p><strong>Company:</strong> {company_name or 'N/A'}</p>
                        <p><strong>Company ID:</strong> {company_id}</p>
                        {f'<p><strong>Job ID:</strong> {lead_run_id}</p>' if lead_run_id else ''}
                        {f'<p><strong>Error:</strong> {error_message}</p>' if error_message else ''}
                    </div>
                    
                    <h3 style="color:#333333; font-size:16px;">Possible Causes:</h3>
                    <ul style="color:#555555; font-size:16px; text-align:left; line-height:1.8;">
                        <li>AI service may be temporarily unavailable</li>
                        <li>Insufficient data extracted from website</li>
                        <li>Processing timeout</li>
                        <li>Data format issues</li>
                    </ul>
                    
                    <h3 style="color:#333333; font-size:16px;">Next Steps:</h3>
                    <ul style="color:#555555; font-size:16px; text-align:left; line-height:1.8;">
                        <li>Try triggering the assessment again</li>
                        <li>Contact support for assistance</li>
                    </ul>
                    
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    <strong>Need help?</strong> Contact us at 
                    <a href="mailto:dev-aws@TheDisruptionHouse.com" style="color:#008087;">
                    dev-aws@TheDisruptionHouse.com</a>
                    </p>
        """
    
    else:
        subject = f"ESG Assessment Notification - {company_name or 'Your Company'}"
        body_content = f"""
                    <p style="color:#555555; font-size:16px; margin:20px 0;">
                    This is a notification regarding your ESG assessment.
                    </p>
        """
    
    html_body = html_header + body_content + html_footer
    
    return subject, html_body

######## AVERAGE TOPIC AND CATEGORY SCORES FUNCTIONS ########

def rds_get_average_metrics3_topic_scores(conn):
    """
    Fetch average topic scores from the view view_metrictopicmeanmedian.
    
    :param conn: A psycopg2 database connection object.
    :return: DataFrame containing average topic scores.
    """
    query = """
    SELECT "TopicDescription", "MeanTopicScore", "MedianTopicScore" FROM view_assessmenttopicscore WHERE "SurveyID" = 3333 
    GROUP BY "TopicDescription", "MeanTopicScore", "MedianTopicScore"
    """
    
    try:
        results = pd.read_sql_query(query, conn)
        print("Successfully fetched average topic scores.")
        return results
    except Exception as e:
        print(f"Error fetching average topic scores: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error
    
def rds_get_average_metrics3_category_scores(conn):
    """
    Fetch average category scores from the view view_metriccategorymeanmedian.
    
    :param conn: A psycopg2 database connection object.
    :return: DataFrame containing average topic scores.
    """
    query = """
    SELECT "CategoryDescription", "MeanCategoryScore", "MedianCategoryScore" FROM view_assessmentcategoryscore  WHERE "SurveyID" = 3333   
    GROUP BY "CategoryDescription", "MeanCategoryScore", "MedianCategoryScore" 
    """

    try:
        results = pd.read_sql_query(query, conn)
        print("Successfully fetched average topic scores.")
        return results
    except Exception as e:
        print(f"Error fetching average topic scores: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error
    

def rds_get_sector_benchmark_stats(conn, sector_id):
    """
    Get sector benchmark statistics (average, min, max, count, label) from CompanyAssessment.
    
    :param conn: A psycopg2 database connection object.
    :param sector_id: Integer sector ID.
    :return: Dict with keys: sectorAverage, sectorMin, sectorMax, scoreCount, sectorLabel.
             Returns dict with zeros if no data found.
    """
    # Use view_benchmarkmetriccategory, which is already constrained to latest snapshot,
    # to derive sector-level benchmark dials from benchmark metric data.
    query = """
        SELECT 
            AVG(bmc."Mean") AS sectorAverage,
            MIN(bmc."Mean") AS sectorMin,
            MAX(bmc."Mean") AS sectorMax,
            COUNT(*) AS scoreCount,
            MAX(bmc."Sector") AS sectorLabel
        FROM "view_benchmarkmetriccategory" bmc
        WHERE bmc."SurveyID" = 3333
          AND bmc."SectorID" = %s
        GROUP BY bmc."SectorID";
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (sector_id,))
            row = cursor.fetchone()
            if row:
                # Map by position to ensure expected camelCase keys
                sector_average, sector_min, sector_max, score_count, sector_label = row
                return {
                    "sectorAverage": float(sector_average) if sector_average is not None else 0,
                    "sectorMin": float(sector_min) if sector_min is not None else 0,
                    "sectorMax": float(sector_max) if sector_max is not None else 100,
                    "scoreCount": score_count or 0,
                    "sectorLabel": sector_label or "Sector Average",
                }
            else:
                return {
                    "sectorAverage": 0,
                    "sectorMin": 0,
                    "sectorMax": 100,
                    "scoreCount": 0,
                    "sectorLabel": "Unknown",
                }
    except Exception as e:
        logging.error(f"Error fetching sector benchmark stats: {e}")
        return {"sectorAverage": 0, "sectorMin": 0, "sectorMax": 100, "scoreCount": 0, "sectorLabel": "Unknown"}


def rds_get_industry_benchmark_stats(conn, industry_id):
    """
    Get industry benchmark statistics (average, min, max, count, label) from CompanyAssessment.
    
    :param conn: A psycopg2 database connection object.
    :param industry_id: Integer industry ID.
    :return: Dict with keys: industryAverage, industryMin, industryMax, scoreCount, industryLabel.
             Returns dict with zeros if no data found.
    """
    # Use view_assessmentcompanyscore to avoid re-aggregating category scores
    query = """
        SELECT 
            AVG(v."CompanySectorScore") AS industryAverage,
            MIN(v."CompanySectorScore") AS industryMin,
            MAX(v."CompanySectorScore") AS industryMax,
            COUNT(*) AS scoreCount,
            i."IndustryDescription" AS industryLabel
        FROM view_assessmentcompanyscore v
        JOIN "Company" c ON v."CompanyID" = c."CompanyID"
        JOIN "Industry" i ON c."IndustryID" = i."IndustryID"
        WHERE v."SurveyID" = 3333
          AND c."IndustryID" = %s
          AND v."CreateDate" >= CURRENT_DATE - INTERVAL '365 days'
        GROUP BY i."IndustryDescription";
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (industry_id,))
            row = cursor.fetchone()
            if row:
                industry_average, industry_min, industry_max, score_count, industry_label = row
                return {
                    "industryAverage": float(industry_average) if industry_average is not None else 0,
                    "industryMin": float(industry_min) if industry_min is not None else 0,
                    "industryMax": float(industry_max) if industry_max is not None else 100,
                    "scoreCount": score_count or 0,
                    "industryLabel": industry_label or "Industry Average",
                }
            else:
                return {
                    "industryAverage": 0,
                    "industryMin": 0,
                    "industryMax": 100,
                    "scoreCount": 0,
                    "industryLabel": "Unknown",
                }
    except Exception as e:
        logging.error(f"Error fetching industry benchmark stats: {e}")
        return {"industryAverage": 0, "industryMin": 0, "industryMax": 100, "scoreCount": 0, "industryLabel": "Unknown"}


def rds_get_revenue_range_benchmarks(conn, sector_id):
    """
    Get revenue range benchmarks from view_benchmarkmetriccategory.
    
    :param conn: A psycopg2 database connection object.
    :param sector_id: Integer sector ID.
    :return: List of dicts with keys: revenueRange, minScore, maxScore, avgScore.
    """
    # Use view_benchmarkmetriccategory which is already constrained to latest snapshot
    query = """
        SELECT 
            bmc."Metric Revenue" AS revenueRange,
            MIN(bmc."Mean") AS minScore,
            MAX(bmc."Mean") AS maxScore,
            AVG(bmc."Mean") AS avgScore
        FROM "view_benchmarkmetriccategory" bmc
        WHERE bmc."SurveyID" = 3333
          AND bmc."SectorID" = %s
          AND bmc."Metric Revenue" IS NOT NULL
        GROUP BY bmc."Metric Revenue"
        ORDER BY bmc."Metric Revenue";
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (sector_id,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                revenue_range, min_score, max_score, avg_score = row
                results.append(
                    {
                        "revenueRange": revenue_range or "",
                        "minScore": float(min_score) if min_score is not None else 0,
                        "maxScore": float(max_score) if max_score is not None else 100,
                        "avgScore": float(avg_score) if avg_score is not None else 0,
                    }
                )
            return results
    except Exception as e:
        logging.error(f"Error fetching revenue range benchmarks: {e}")
        return []


def rds_get_employee_range_benchmarks(conn, sector_id):
    """
    Get employee range benchmarks from view_benchmarkmetriccategory.
    
    :param conn: A psycopg2 database connection object.
    :param sector_id: Integer sector ID.
    :return: List of dicts with keys: employeeRange, minScore, maxScore, avgScore.
    """
    # Use view_benchmarkmetriccategory which is already constrained to latest snapshot
    query = """
        SELECT 
            bmc."Number of Employee" AS employeeRange,
            MIN(bmc."Mean") AS minScore,
            MAX(bmc."Mean") AS maxScore,
            AVG(bmc."Mean") AS avgScore
        FROM "view_benchmarkmetriccategory" bmc
        WHERE bmc."SurveyID" = 3333
          AND bmc."SectorID" = %s
          AND bmc."Number of Employee" IS NOT NULL
        GROUP BY bmc."Number of Employee"
        ORDER BY bmc."Number of Employee";
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (sector_id,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                employee_range, min_score, max_score, avg_score = row
                results.append(
                    {
                        "employeeRange": employee_range or "",
                        "minScore": float(min_score) if min_score is not None else 0,
                        "maxScore": float(max_score) if max_score is not None else 100,
                        "avgScore": float(avg_score) if max_score is not None else 0,
                    }
                )
            return results
    except Exception as e:
        logging.error(f"Error fetching employee range benchmarks: {e}")
        return []


def rds_get_word_cloud_terms(conn, sector_id, limit=50):
    """
    Get word cloud terms from URLTerm table for companies in the specified sector.
    
    :param conn: A psycopg2 database connection object.
    :param sector_id: Integer sector ID.
    :param limit: Maximum number of terms to return (default: 50).
    :return: List of dicts with keys: TermID, text, totalCount, companyCount.
    """
    # Use aggregated view_companytermcount to reduce row volume compared to raw URLTerm
    # Still filter by sector and recency via LeadRun.EndDate
    query = """
        SELECT
            term."TermID",
            term."TermDescription" AS text,
            SUM(v."TotalCompanyTermCount") AS totalCount,
            COUNT(DISTINCT v."CompanyID") AS companyCount
        FROM view_companytermcount v
        JOIN "Company" c ON c."CompanyID" = v."CompanyID"
        JOIN "LeadRun" lr ON lr."LeadRunID" = v."LeadRunID"
        JOIN "Term" term ON term."TermID" = v."TermID"
        WHERE c."SectorID" = %s
          AND lr."EndDate" >= CURRENT_DATE - INTERVAL '365 days'
        GROUP BY term."TermID", term."TermDescription"
        ORDER BY totalCount DESC
        LIMIT %s;
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (sector_id, limit))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                term_id, text, total_count, company_count = row
                results.append(
                    {
                        "TermID": term_id,
                        "text": text or "",
                        "totalCount": total_count or 0,
                        "companyCount": company_count or 0,
                    }
                )
            return results
    except Exception as e:
        logging.error(f"Error fetching word cloud terms: {e}")
        return []


# -------------------------------
# Check if company is in CompanyScrape
def check_company_in_scrape_table(conn, company_id):
    with conn.cursor() as cursor:
        query = 'SELECT "CompanyID" FROM "CompanyScrape" WHERE "CompanyID" = %s'
        cursor.execute(query, (company_id,))
        return cursor.fetchone() is not None

# -------------------------------
# Get valid company URL
def get_company_url(conn, company_id):
    with conn.cursor() as cursor:
        query = 'SELECT "CompanyWebSite" FROM "Company" WHERE "CompanyID" = %s'
        cursor.execute(query, (company_id,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else None

# -------------------------------
# Insert a record into LeadRun table
def insert_lead_run(conn, company_id, session_id, company_url, worker_group):
    with conn.cursor() as cursor:
        query = '''
            INSERT INTO "LeadRun" ("CompanyID", "KadoaSessionID", "StartDate", "MainURL", "ProcessStatus", "WorkerGroup")
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING "LeadRunID"
        '''
        cursor.execute(query, (company_id, session_id, datetime.now(), company_url, 9, worker_group))
        lead_run_id = cursor.fetchone()[0]
        conn.commit()
        return lead_run_id

# -------------------------------
# Insert a record into LeadRunHistory table
def insert_lead_run_history(conn, lead_run_id, process):
    with conn.cursor() as cursor:
        query = '''
            INSERT INTO "LeadRunHistory" ("LeadRunID", "StartDate", "Process")
            VALUES (%s, %s, %s)
            RETURNING "LeadRunHistoryID"
        '''
        cursor.execute(query, (lead_run_id, datetime.now(), process))
        lead_run_history_id = cursor.fetchone()[0]
        conn.commit()
        return lead_run_history_id

# -------------------------------
# Update LeadRunHistory with end date and optional process message
def update_lead_run_history_end(conn, lead_run_history_id, process=None):
    with conn.cursor() as cursor:
        if process:
            query = '''
                UPDATE "LeadRunHistory"
                SET "EndDate" = %s, "Process" = %s
                WHERE "LeadRunHistoryID" = %s
            '''
            params = (datetime.now(), process, lead_run_history_id)
        else:
            query = '''
                UPDATE "LeadRunHistory"
                SET "EndDate" = %s
                WHERE "LeadRunHistoryID" = %s
            '''
            params = (datetime.now(), lead_run_history_id)

        cursor.execute(query, params)
        conn.commit()

# -------------------------------
# Update LeadRun table with end date and process status
def update_lead_run_status(conn, lead_run_id, process_status):
    with conn.cursor() as cursor:
        query = '''
            UPDATE "LeadRun"
            SET "EndDate" = %s, "ProcessStatus" = %s
            WHERE "LeadRunID" = %s
        '''
        cursor.execute(query, (datetime.now(), process_status, lead_run_id))
        conn.commit()

# -------------------------------
# Poll Kadoa API until job is finished
def wait_for_kadoa_job(session_id, api_key, timeout_seconds=1500):
    start_time = time.time()
    while True:
        status = kadoa.kadoa_get_job_status(session_id, api_key)
        print("Status:", status)
        if status == "job finished":
            return "job finished"
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"Timeout: Kadoa job did not finish within {timeout_seconds / 60} minutes.")
            return "timeout"
        time.sleep(5)

# -------------------------------
# Save results and return file count
def scrape_company_website(company_name, max_pages, session_id, api_key):
    base_dir = Path("/tmp")
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', company_name)
    folder_name = f"RAW_{safe_name}_{max_pages}"
    save_path = base_dir / folder_name

    kadoa.kadoa_save_results(session_id, api_key, str(save_path), False)

    if not save_path.exists():
        return None
    files = list(save_path.glob("*.txt"))
    return len(files)


# Save L1 score in the Lead Run table
def update_lead_run_l1_score(conn, lead_run_id, raw_score):
    """
    Update the L1 score in the LeadRun table.
    
    :param conn: A psycopg2 database connection object.
    :param lead_run_id: The LeadRunID to update.
    :param raw_score: The L1 score to save.
    """

    score_percentage = (raw_score / 24) * 100
    with conn.cursor() as cursor:
        query = '''
            UPDATE "LeadRun"
            SET "ScoreAbsolute" = %s, "ScorePercentage" = %s
            WHERE "LeadRunID" = %s
        '''
        cursor.execute(query, (raw_score,score_percentage, lead_run_id))
        conn.commit()

# -------------------------------

# Main Function for L1 scoring of multiple companies
def scrape_multiple_companies(conn, companies, worker_group, env, secrets, max_pages=300, scoring_level=None):
    companies_not_in_scrape = []
    companies_with_invalid_urls = []
    kadoa_api_key = "f23a05b4-e922-462c-bf4a-09e2c169764f" 

    base_dir = Path("/tmp")
    save_dir = None
    base_folder = None
    clean_pdfs_path = None
    session_id = None
    lead_run_id = None
    df_info = None

    all_results = []

    for _, row in companies.iterrows():

        company_id = row["CompanyID"]
        company_name = row["CompanyName"]
        company_secret = row["Secret"]  # Assuming this is in the DataFrame
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', company_name)
        path = base_dir / f"{safe_name}_{max_pages}"

        print(f"{company_name} has ID {company_id}")

        # Step 1: Check if company is in CompanyScrape table
        # if not check_company_in_scrape_table(conn, company_id):
        #     print(f"Γ¥î {company_name} not in CompanyScrape. Skipping.")
        #     companies_not_in_scrape.append(company_name)
        #     continue

        # Step 2: Check if company has a valid URL
        company_url = get_company_url(conn, company_id)
        if not company_url:
            print(f"Γ¥î {company_name} has no valid URL. Skipping.")
            companies_with_invalid_urls.append(company_name)
            continue

        # Skip if folder already exists
        if path.exists():
            print(f"Γ£à Folder {path} exists. Skipping scrape.")
            print("-" * 100)
            continue

        # --------- Wrapped in two-attempts (retry-on-error) block ---------
        scrape_attempts = 0
        max_scrape_attempts = 2
        retry = True
        lead_run_id = None  # Initialize outside loop to persist across retries
        while retry and scrape_attempts < max_scrape_attempts:
            retry = False
            try:
                t = time.time()
                print(f"≡ƒÜÇ Starting scrape for {company_name} ({company_url}) [Attempt {scrape_attempts+1}]")

                # Step 3.1: Start the crawl job
                session_id = kadoa.kadoa_start_job(company_url, kadoa_api_key, max_pages=max_pages)
                if not session_id:
                    print(f"Γ¥î Failed to start Kadoa job for {company_name}.")
                    break

                # Step 3.1.1: Get or create the Lead Run ID (reuse on retry)
                if lead_run_id is None:
                    # First attempt: try to find existing or create new
                    with conn.cursor() as cursor:
                        query = '''
                            SELECT "LeadRunID" FROM "LeadRun"
                            WHERE "CompanyID" = %s AND "WorkerGroup" = %s
                        '''
                        cursor.execute(query, (company_id, worker_group))
                        lead_run_row = cursor.fetchone()
                        if lead_run_row:
                            lead_run_id = lead_run_row[0]
                            print(f"Γ£à Found existing LeadRunID: {lead_run_id}")
                        else:
                            print(f"≡ƒô¥ Creating new LeadRun for {company_name}.")
                            lead_run_id = insert_lead_run(conn, company_id, session_id, company_url, worker_group)
                else:
                    # Retry attempt: reuse the same LeadRunID
                    print(f"≡ƒöä Retry attempt: Reusing LeadRunID: {lead_run_id}")
                
                # Update the Kadoa session ID in the LeadRun table
                with conn.cursor() as cursor:
                    query = ''' 
                        UPDATE "LeadRun"
                        SET "KadoaSessionID" = %s
                        WHERE "LeadRunID" = %s
                    '''
                    cursor.execute(query, (session_id, lead_run_id))
                    conn.commit()

                # Step 3.1.2: Insert LeadRunHistory (waiting)
                history_id_waiting = insert_lead_run_history(conn, lead_run_id, "Kadoa job running, waiting for results")

                # Step 3.2: Wait for Kadoa job to finish
                result = wait_for_kadoa_job(session_id, kadoa_api_key)
                if result == "timeout":
                    update_lead_run_status(conn, lead_run_id, 3)  # Update status to 3 ("Error Processing")
                    update_lead_run_history_end(conn, history_id_waiting)
                    
                    history_id_timeout = insert_lead_run_history(conn, lead_run_id, "Kadoa job timed out, no pages saved")
                    update_lead_run_history_end(conn, history_id_timeout)
                    break  # no retry for timeout
                elif result == "job finished":
                    print("Proceeding with next steps.")

                # Step 3.2.1: Insert LeadRunHistory (job finished)
                history_id_job_done_saving_results = insert_lead_run_history(conn, lead_run_id, "Kadoa job finished, now saving results")
                update_lead_run_history_end(conn, history_id_waiting)

                # Step 3.2.2: Update LeadRun process status = 2 (saving results)
                update_lead_run_status(conn, lead_run_id, 2)

                # Step 3.3: Save results and count files
                kadoa_results = kadoa.kadoa_get_job_results(session_id, kadoa_api_key)
                total_pages_scraped = kadoa_results["pagination"]["totalItems"]
                print(f"Total pages scraped: {total_pages_scraped}")
                if total_pages_scraped is None:
                    all_results.append({"company_id": company_id, "company_name": company_name, "error": "Folder not found"})
                    break
                elif total_pages_scraped < 2:
                    process_status_id = 4
                else:
                    process_status_id = 5

                # Step 3.3.1: Update LeadRunHistory end date
                update_lead_run_history_end(conn, history_id_job_done_saving_results)

                # Step 3.3.2: Insert LeadRunHistory for processing results
                history_id_processing_results = insert_lead_run_history(conn, lead_run_id, "Processing results")

                # Step 4: Score the results
                save_dir = f"/tmp/RAW_{safe_name}_{max_pages}"

                results =  kadoa.kadoa_process_results(
                    session_id, 
                    kadoa.kadoa_get_job_results(session_id, kadoa_api_key), 
                    kadoa_api_key, 
                    conn, 
                    lead_run_id, 
                    save_dir,
                    clean=False
                )

                # print(results)

                # Step 4.1: Update LeadRun with total score
                total_score_value = int(results["total_score"]["Score"].sum())
                update_lead_run_l1_score(conn, lead_run_id, total_score_value)

                # Step 4.2: Update LeadRunHistory with end date for processing results
                update_lead_run_history_end(conn, history_id_processing_results)

                # Step 4.3: Insert LeadRunHistory for uploading to S3
                history_id_uploading_to_s3 = insert_lead_run_history(conn, lead_run_id, "Uploading pages to S3")

                # Step 4.4: Get documents and upload to S3
                download_company_documents(save_dir, company_name, company_secret, company_url, env, secrets=secrets)

                # Step 5: Push the pages to S3
                base_folder = Path("/tmp") / f"RAW_{safe_name}_{max_pages}"
                uploaded_count, s3_folder_link = kadoa.upload_company_pages_to_s3(
                    metadata=results["metadata"],
                    company_secret=company_secret,
                    env=env,
                    base_folder=base_folder,
                    scrape_date=datetime.today().strftime("%Y-%m-%d")
                )

                if uploaded_count <= 2:
                    process_status_id = 4  # Blocked or scraped fewer than 2 pages
                else:
                    process_status_id = 5  # Successfully scraped

                # Step 5.1: Update LeadRun table with S3 link
                with conn.cursor() as cursor:
                    query = '''
                        UPDATE "LeadRun"
                        SET "S3FolderLink" = %s
                        WHERE "LeadRunID" = %s
                    '''
                    cursor.execute(query, (s3_folder_link, lead_run_id))
                    conn.commit()

                # Step 5.2: Update LeadRunHistory with end date for uploading to S3
                update_lead_run_history_end(conn, history_id_uploading_to_s3)

                # Step 5.3: Update LeadRun with final process status
                update_lead_run_status(conn, lead_run_id, process_status_id)

                # Step 6: Updated LeadRunHistory with final 'Completed' process
                history_id_completed = insert_lead_run_history(conn, lead_run_id, "Completed")
                update_lead_run_history_end(conn, history_id_completed)

                # Optional L1+ scoring path (Metrics L1+, SurveyID = 1111)
                # Only run L1+ if scrape was successful (ProcessStatus = 5)
                if scoring_level == 'L1+':
                    if process_status_id == 5:
                        try:
                            print(f"Γ£¿ Starting L1+ OpenAI scoring for {company_name}...")

                            # Determine sector to use for CompanyAssessment
                            df_info = rds_get_company_information(conn, company_id=company_id, all_columns=True)
                            if df_info.empty or pd.isna(df_info.get("SectorID", [None])[0]):
                                company_score_sector_id = None
                            else:
                                sector_val = df_info["SectorID"].iloc[0]
                                company_score_sector_id = to_native(sector_val) if sector_val is not None else None

                            # Create CompanyAssessment for L1+ (survey 1111)
                            companyassessment_id = rds_create_company_assessment_id(
                                conn,
                                company_id,
                                survey_id=1111,
                                evaluation_type_id=7,
                                company_score_sector_id=company_score_sector_id,
                            )
                            print(f"Γ£à Created CompanyAssessmentID: {companyassessment_id} for {company_name} (SurveyID=1111)")

                            # Skip PDF generation, use chunking/embedding approach directly with .txt files
                            print(f"≡ƒôÜ Using chunking/embedding approach for L1+ scoring (skipping PDF generation)")

                            # Run L1+ scoring via shared ESG pipeline with chunking
                            print(f"≡ƒñû Generating L1+ score for {company_name} (AssessmentID: {companyassessment_id})")
                            esg.generate_m3_score(
                                conn, 
                                companyassessment_id, 
                                save_dir,  # Pass .txt files folder directly (not PDFs)
                                secrets=secrets,
                                scoring_level='L1+',
                                company_name=company_name,
                                use_chunking=True  # Enable chunking mode
                            )

                            # Update category and topic scores
                            esg.rds_insert_or_update_category_scores(conn, companyassessment_id)
                            esg.rds_insert_or_update_topic_scores(conn, companyassessment_id)

                            # Recompute & write overall CompanyScore from view.CompanySectorScore (+ sector)
                            print(f"≡ƒº« Recomputing overall company score for AssessmentID {companyassessment_id}")
                            updated_score_df = esg.rds_update_survey_score(conn, companyassessment_id)
                            if updated_score_df is not None and not updated_score_df.empty:
                                print(
                                    f"Γ£ö CompanyAssessmentID {int(updated_score_df.iloc[0]['CompanyAssessmentID'])} "
                                    f"CompanyScore={float(updated_score_df.iloc[0]['CompanyScore'])} "
                                    f"SectorID={int(updated_score_df.iloc[0]['CompanyScoreSectorID'])}"
                                )
                            
                            # Set AssessmentStatusID to 34 (Completed)
                            with conn.cursor() as cursor:
                                cursor.execute('''
                                    UPDATE "CompanyAssessment"
                                    SET "AssessmentStatusID" = 34, 
                                    "SurveyMonkeyModifiedDate" = NOW()
                                    WHERE "CompanyAssessmentID" = %s
                                ''', (companyassessment_id,))
                                conn.commit()
                                print(f"Γ£à Set AssessmentStatusID to 34 for CompanyAssessmentID {companyassessment_id}")

                            # Activate the assessment (marks old ones as Inactive)
                            try:
                                print(f"≡ƒöä Activating CompanyAssessmentID {companyassessment_id}...")
                                activation_result = esg.activate_company_assessment(conn, companyassessment_id, rescore=False)
                                if activation_result:
                                    deactivated_count = len(activation_result.get('deactivated_assessment_ids', []))
                                    if deactivated_count > 0:
                                        print(f"Γ£ö Activated assessment and marked {deactivated_count} previous assessment(s) as Inactive")
                                    else:
                                        print("Γ£ö Activated assessment (no previous active assessments found)")
                            except Exception as activation_error:
                                print(f"ΓÜá∩╕Å Failed to activate assessment: {activation_error}")

                        except Exception as l1_plus_error:
                            print(f"ΓÜá∩╕Å L1+ scoring failed for {company_name}: {l1_plus_error}")
                    else:
                        print(f"ΓÅ¡∩╕Å Skipping L1+ scoring for {company_name} (scrape not successful, ProcessStatus={process_status_id})")

                # Calculate elapsed time
                elapsed = time.time() - t
                print(f"Total time taken: {elapsed / 60:.2f} minutes")
                print("-" * 100)
                
                all_results.append({
                    "company_id": company_id,
                    "company_name": company_name
                    # "result": results
                })
                break  # Success, no retry
            except Exception as e:
                conn.rollback()

                if lead_run_id is not None:
                    try:
                        with conn.cursor() as cursor:
                            query = '''
                                UPDATE "LeadRun"
                                SET "ProcessStatus" = 3, "EndDate" = %s
                                WHERE "LeadRunID" = %s
                            '''
                            cursor.execute(query, (datetime.now(), lead_run_id))
                            conn.commit()
                    except Exception as update_error:
                        print(f"ΓÜá∩╕Å Failed to update LeadRun with error status: {update_error}")
                else:
                    print("ΓÜá∩╕Å lead_run_id not available, skipping DB update for error status")

                print(f"Γ¥î Error processing {company_name} (ID: {company_id}) on attempt {scrape_attempts+1}: {e}")
                print("Continuing to next company...\n")
                
                # If this was the first attempt, retry once
                if scrape_attempts == 0:
                    print(f"≡ƒöä Retrying scrape for {company_name} (ID: {company_id})...")
                    retry = True
                scrape_attempts += 1

        # -------------------- Memory cleanup --------------------
        
        try:
            del results
            del clean_pdfs_path
            del session_id
            del df_info
            del companyassessment_id
        except NameError:
            pass
        
        gc.collect()

        # Remove the temporary folders

        try:
            if save_dir and Path(save_dir).exists():
                shutil.rmtree(save_dir)

            if base_folder and Path(base_folder).exists():
                shutil.rmtree(base_folder)

            cleaned_dir = base_dir / f"{safe_name}_cleaned_pdfs"
            if cleaned_dir.exists():
                shutil.rmtree(cleaned_dir)

            print(f"Temp folders deleted for {company_name}")

        except Exception as folder_error:
            print(f"Failed to delete temp folders for {company_name}: {folder_error}")
            
        print(f"≡ƒº╣ Memory cleanup done for {company_name}")
        print("-" * 100)

    return all_results

def rds_create_company_assessment_id(conn, company_id: int, survey_id: int, evaluation_type_id: int | None, company_score_sector_id: int | None = 12) -> int:
    """
    Create a new CompanyAssessment row and return its ID.

    Mirrors the Plumber endpoint's sector logic:
      - If company_score_sector_id is provided (not None), use it.
      - Else, fall back to Company.SectorID for the given company_id.
      - If both are NULL, insert NULL for CompanyScoreSectorID.

    Notes:
      - AssessmentStatusID is initially set to 18 (as in your current Python).
      - ProductID is set to 28 (as in your current Python).
      - SurveyID and EvaluationTypeID are stored as provided (EvaluationTypeID may be NULL).
    """
    with conn.cursor() as cur:
        # Resolve fallback sector from Company if needed
        effective_sector_id = company_score_sector_id
        if effective_sector_id is None:
            cur.execute(
                'SELECT "SectorID" FROM "Company" WHERE "CompanyID" = %s',
                (company_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"company_id {company_id} not found in Company table.")
            effective_sector_id = row[0]  # may be None

        # Insert CompanyAssessment with sector
        cur.execute(
            """
            INSERT INTO "CompanyAssessment"
                ("CompanyID",
                 "AssessmentStatusID",
                 "ProductID",
                 "SurveyID",
                 "CreateDate",
                 "EvaluationTypeID",
                 "CompanyScoreSectorID",
                 "AssessmentValidStatusID")
            VALUES
                (%s, 18, 28, %s, NOW(), %s, %s, 1)
            RETURNING "CompanyAssessmentID"
            """,
            (company_id, survey_id, evaluation_type_id, effective_sector_id),
        )
        company_assessment_id = cur.fetchone()[0]

        conn.commit()
        return company_assessment_id
    
def generate_m3_scores_multiple_companies(conn, companies, scoring_level, secrets, company_score_sector_id: int | None = 12):
    # Validate scoring level
    if scoring_level not in ['L1+', 'L2', 'L3']:
        print(f"Γ¥î Invalid scoring level: {scoring_level}. Must be 'L1+', 'L2', or 'L3'.")
        return []
    
    base_dir = Path("/tmp")
    all_results = []

    # Initialize variables to avoid NameError during cleanup
    updated_score_df = None
    clean_pdfs_path = None
    companyassessment_id = None
    activation_result = None
    history_id_processing = None
    history_id_scored = None
    history_id_completed = None

    for _, row in companies.iterrows():
        company_id = row["CompanyID"]
        company_name = row["CompanyName"]
        s3_folder_link = row["S3FolderLink"]
        secret = row["Secret"]
        end_date = row["EndDate"]
        lead_run_id = row["LeadRunID"]

        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', company_name)
        local_path = base_dir / f"{safe_name}_S3_pages"
        cleaned_pdfs_dir = base_dir / f"{safe_name}_cleaned_pdfs"

        try:
            t = time.time()
            print(f"≡ƒÜÇ Starting {scoring_level} scoring for {company_name} (ID: {company_id})")
            print(f"S3 Folder Link: {s3_folder_link}")
            print(f"Secret: {secret}")
            print(f"End Date: {end_date}")

            # Step 1: Create a CompanyAssessment entry
            # Use SurveyID 1111 for L1+ (Metrics L1+), otherwise default to 3333 (Metrics L2/L3)
            survey_id = 1111 if scoring_level == 'L1+' else 3333
            companyassessment_id = rds_create_company_assessment_id(
                conn, company_id, survey_id=survey_id, evaluation_type_id=7, company_score_sector_id=company_score_sector_id
            )
            print(f"Γ£à Created CompanyAssessmentID: {companyassessment_id} for {company_name} (SurveyID={survey_id})")

            # Step 2: Update LeadRun with ProcessStatus 6 (Processing)
            update_lead_run_status(conn, lead_run_id, 6)

            # Step 3: Insert LeadRunHistory for processing
            history_id_processing = insert_lead_run_history(conn, lead_run_id, f"Starting {scoring_level} scoring process")
            print(f"≡ƒòÆ Inserted LeadRunHistory for processing: {history_id_processing}")

            # Step 4: Download pages from S3
            if not s3_folder_link:
                print(f"Γ¥î {company_name} has no S3 folder link. Skipping.")
                update_lead_run_history_end(conn, history_id_processing)
                update_lead_run_status(conn, lead_run_id, 7)

                error_processing_id = insert_lead_run_history(conn, lead_run_id, f"{scoring_level} error: No S3 folder link")
                update_lead_run_history_end(conn, error_processing_id)
                continue

            print(f"≡ƒôÑ Downloading pages for {company_name} from S3...")
            esg.download_pages_from_s3(s3_folder_link, local_path, secrets=secrets)

            if not os.path.exists(local_path) or len(os.listdir(local_path)) < 2:
                print(f"ΓÜá∩╕Å {company_name} has fewer than 2 pages. Skipping {scoring_level} scoring.")
                update_lead_run_history_end(conn, history_id_processing)
                update_lead_run_status(conn, lead_run_id, 7)  # Blocked or insufficient data
                
                less_than_2_pages_id = insert_lead_run_history(conn, lead_run_id, f"{scoring_level} error: Fewer than 2 pages")
                update_lead_run_history_end(conn, less_than_2_pages_id)

                continue

            # Step 5: Skip PDF generation for L2, use chunking/embedding approach directly with .txt files
            if scoring_level == 'L2':
                print(f"≡ƒôÜ Using chunking/embedding approach for L2 scoring (skipping PDF generation)")
                # Step 6: Score using chunking
                print(f"≡ƒñû Generating {scoring_level} score for {company_name} (AssessmentID: {companyassessment_id})")
                esg.generate_m3_score(
                    conn, 
                    companyassessment_id, 
                    local_path,  # Pass .txt files folder directly (not PDFs)
                    secrets=secrets,
                    scoring_level=scoring_level,
                    company_name=company_name,
                    use_chunking=True  # Enable chunking mode
                )
            else:
                # For L3, still use PDF generation (chunking not supported for L3)
                print(f"≡ƒº╣ Cleaning pages and converting to PDF for {company_name}")
                clean_pdfs_path = kadoa.generate_cleaned_pdf_batches(
                    company_name=company_name,
                    folder_path=local_path,
                    output_folder=cleaned_pdfs_dir,
                    token_limit=500_000,
                    max_pages=500,
                )
                # Step 6: Score the cleaned PDFs
                print(f"≡ƒñû Generating {scoring_level} score for {company_name} (AssessmentID: {companyassessment_id})")
                esg.generate_m3_score(conn, companyassessment_id, clean_pdfs_path, scoring_level=scoring_level, secrets=secrets)

            # Step 7: Update LeadRun with ProcessStatus 8 (Completed)
            update_lead_run_status(conn, lead_run_id, 8)

            # Step 8: Update LeadRunHistory
            update_lead_run_history_end(conn, history_id_processing)
            history_id_scored = insert_lead_run_history(conn, lead_run_id, f"{scoring_level} score generated and saved to RDS")
            update_lead_run_history_end(conn, history_id_scored)

            # Step 9: Update CompanyAssessment status
            with conn.cursor() as cursor:
                query = """
                    UPDATE public."CompanyAssessment"
                    SET "AssessmentStatusID" = 34, "SurveyMonkeyModifiedDate" = NOW()
                    WHERE "CompanyAssessmentID" = %s;
                """
                cursor.execute(query, (companyassessment_id,))
                conn.commit()

            # Step 10: Insert LeadRunHistory for completion
            history_id_completed = insert_lead_run_history(conn, lead_run_id, f"{scoring_level} completed")
            update_lead_run_history_end(conn, history_id_completed)

            # Step 11: Update Category Scores
            esg.rds_insert_or_update_category_scores(conn, companyassessment_id)

            # Step 12: Update Topic Scores
            esg.rds_insert_or_update_topic_scores(conn, companyassessment_id)

            # Step 13: Recompute & write overall CompanyScore from view.CompanySectorScore (+ sector)
            #         (uses your updated rds_update_survey_score implementation)
            print(f"≡ƒº« Recomputing overall company score for AssessmentID {companyassessment_id}")
            updated_score_df = esg.rds_update_survey_score(conn, companyassessment_id)
            if updated_score_df is not None and not updated_score_df.empty:
                print(
                    f"Γ£ö CompanyAssessmentID {int(updated_score_df.iloc[0]['CompanyAssessmentID'])} "
                    f"CompanyScore={float(updated_score_df.iloc[0]['CompanyScore'])} "
                    f"SectorID={int(updated_score_df.iloc[0]['CompanyScoreSectorID'])}"
                )
            else:
                print("ΓÜá∩╕Å No updated overall score returned by rds_update_survey_score")

            # Step 13b: Activate the assessment (marks old ones as Inactive)
            try:
                print(f"≡ƒöä Activating CompanyAssessmentID {companyassessment_id}...")
                activation_result = esg.activate_company_assessment(conn, companyassessment_id, rescore=False)
                if activation_result:
                    deactivated_count = len(activation_result.get('deactivated_assessment_ids', []))
                    if deactivated_count > 0:
                        print(f"Γ£ö Activated assessment and marked {deactivated_count} previous assessment(s) as Inactive")
                    else:
                        print("Γ£ö Activated assessment (no previous active assessments found)")
            except Exception as activation_error:
                print(f"ΓÜá∩╕Å Failed to activate assessment: {activation_error}")

            # Step 15: Track result
            elapsed = time.time() - t
            print(f"Γ£à {scoring_level} scoring completed for {company_name} in {elapsed / 60:.2f} minutes")
            print("-" * 100)
            all_results.append({
                "company_id": company_id,
                "company_name": company_name,
                "assessment_id": companyassessment_id,
                "updated_company_score": (
                    updated_score_df.to_dict(orient="records")[0]
                    if updated_score_df is not None and not updated_score_df.empty
                    else None
                ),
            })

        except Exception as e:
            conn.rollback()
            if lead_run_id is not None:
                try:
                    with conn.cursor() as cursor:
                        query = '''
                            UPDATE "LeadRun"
                            SET "ProcessStatus" = 7, "EndDate" = %s
                            WHERE "LeadRunID" = %s
                        '''
                        cursor.execute(query, (datetime.now(), lead_run_id))
                        conn.commit()
                except Exception as update_error:
                    print(f"ΓÜá∩╕Å Failed to update LeadRun with error status: {update_error}")
            print(f"Γ¥î Error processing {company_name} (ID: {company_id}): {e}")
            print("Continuing to next company...\n")

        finally:
            # -------------------- Memory cleanup --------------------
            try:
                del updated_score_df
                del clean_pdfs_path
                del companyassessment_id
                del activation_result
                del history_id_processing
                del history_id_scored
                del history_id_completed
            except NameError:
                pass

            gc.collect()

            # -------------------- Disk cleanup --------------------
            try:
                if local_path and Path(local_path).exists():
                    shutil.rmtree(local_path)

                if cleaned_pdfs_dir and Path(cleaned_pdfs_dir).exists():
                    shutil.rmtree(cleaned_pdfs_dir)

                print(f"≡ƒº╣ Temp folders deleted for {company_name}")

            except Exception as folder_error:
                print(f"ΓÜá∩╕Å Failed to delete temp folders for {company_name}: {folder_error}")

            print(f"≡ƒº╣ Memory cleanup done for {company_name}")
            print("-" * 100)

    return all_results

def scrape_single_company(conn, company_id, worker_group, env, secrets, max_pages=300, scoring_level=None, lead_run_id=None):
    base_dir = Path("/tmp")
    kadoa_api_key = "f23a05b4-e922-462c-bf4a-09e2c169764f"

    # Fetch company details
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute('''
            SELECT "CompanyID", "CompanyName", "Secret", "CompanyWebSite" FROM "Company" WHERE "CompanyID" = %s
        ''', (company_id,))
        company_row = cursor.fetchone()

    company_name = company_row["CompanyName"]
    company_secret = company_row["Secret"]
    company_url = company_row["CompanyWebSite"]

    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', company_name)
    path = base_dir / f"{safe_name}_{max_pages}"

    max_retries = 2  # maximum number of tries total (first + one retry)
    attempts = 0
    lead_run_id = None  # Initialize outside loop to persist across retries
    while attempts < max_retries:
        attempts += 1
        try:
            t = time.time()

            print(f"{company_name} has ID {company_id}")

            # Step 1: Check if company has a valid URL
            company_url_refresh = get_company_url(conn, company_id)
            if not company_url_refresh:
                print(f"Γ¥î {company_name} has no valid URL. Skipping.")
                return {"company_id": company_id, "company_name": company_name, "error": "No valid URL"}

            # Step 2: Skip if folder already exists
            if path.exists():
                print(f"Γ£à Folder {path} exists. Skipping scrape.")
                print("-" * 100)
                return {"company_id": company_id, "company_name": company_name, "status": "Folder already exists"}

            print(f"≡ƒÜÇ Starting scrape for {company_name} ({company_url_refresh}) [Attempt {attempts}]")

            # Step 3.1: Use provided lead_run_id or create/find one
            if lead_run_id is None:
                # First attempt: try to find existing or create new
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT "LeadRunID" FROM "LeadRun"
                        WHERE "CompanyID" = %s AND "WorkerGroup" = %s AND "ProcessStatus" = 9
                    ''', (company_id, worker_group))
                    lead_run_row = cursor.fetchone()
                    if lead_run_row:
                        lead_run_id = lead_run_row[0]
                        print(f"Γ£à Found existing LeadRunID: {lead_run_id}")
                    else:
                        print(f"≡ƒô¥ Creating new LeadRun for {company_name}.")
                        # Create LeadRun with None for session_id (will be updated when job starts successfully)
                        # We need to create it before starting Kadoa job so we can track errors
                        lead_run_id = insert_lead_run(conn, company_id, None, company_url_refresh, worker_group)
            else:
                # LeadRunID was provided (from trigger-assessment endpoint)
                print(f"Γ£à Using provided LeadRunID: {lead_run_id}")

            # Step 3.2: Start the crawl job (now that we have lead_run_id for error tracking)
            try:
                session_id = kadoa.kadoa_start_job(company_url_refresh, kadoa_api_key, max_pages=max_pages)
            except (ValueError, KeyError) as kadoa_error:
                # Handle Kadoa API errors (e.g., insufficient credits)
                error_msg = str(kadoa_error)
                print(f"Γ¥î Failed to start Kadoa job for {company_name}: {error_msg}")
                
                # Update LeadRun with error status since we have lead_run_id
                if lead_run_id:
                    try:
                        update_lead_run_status(conn, lead_run_id, 3)  # 3 = Error Processing
                        insert_lead_run_history(conn, lead_run_id, f"Kadoa job failed: {error_msg}")
                    except Exception:
                        pass
                
                return {"company_id": company_id, "company_name": company_name, "error": f"Kadoa start failed: {error_msg}"}
            
            if not session_id:
                print(f"Γ¥î Failed to start Kadoa job for {company_name}.")
                # Update LeadRun with error status
                if lead_run_id:
                    try:
                        update_lead_run_status(conn, lead_run_id, 3)  # 3 = Error Processing
                        insert_lead_run_history(conn, lead_run_id, "Kadoa job failed: No session ID returned")
                    except Exception:
                        pass
                return {"company_id": company_id, "company_name": company_name, "error": "Kadoa start failed"}
            
            # Update the Kadoa session ID in the LeadRun table
            with conn.cursor() as cursor:
                cursor.execute('''
                    UPDATE "LeadRun"
                    SET "KadoaSessionID" = %s
                    WHERE "LeadRunID" = %s
                ''', (session_id, lead_run_id))
                conn.commit()

            # Insert LeadRunHistory (waiting)
            history_id_waiting = insert_lead_run_history(conn, lead_run_id, "Kadoa job running, waiting for results")
            
            # Update ProcessStatus to 1 ("Started Scrape") now that scraping has begun
            # Note: We update ProcessStatus without EndDate since scraping is just starting
            with conn.cursor() as cursor:
                cursor.execute('''
                    UPDATE "LeadRun"
                    SET "ProcessStatus" = %s
                    WHERE "LeadRunID" = %s
                ''', (1, lead_run_id))
                conn.commit()

            # Step 3.2: Wait for Kadoa job
            result = wait_for_kadoa_job(session_id, kadoa_api_key)
            if result == "timeout":
                update_lead_run_status(conn, lead_run_id, 3)
                update_lead_run_history_end(conn, history_id_waiting)
                history_id_timeout = insert_lead_run_history(conn, lead_run_id, "Kadoa job timed out, no pages saved")
                update_lead_run_history_end(conn, history_id_timeout)
                return {"company_id": company_id, "company_name": company_name, "error": "Kadoa timeout"}

            update_lead_run_history_end(conn, history_id_waiting)
            history_id_job_done = insert_lead_run_history(conn, lead_run_id, "Kadoa job finished, now saving results")
            update_lead_run_status(conn, lead_run_id, 2)

            # Step 3.3: Save results and count files

            kadoa_results = kadoa.kadoa_get_job_results(session_id, kadoa_api_key)
            total_pages_scraped = kadoa_results["pagination"]["totalItems"]
            print(f"Total pages scraped: {total_pages_scraped}")
            if total_pages_scraped is None:
                return {"company_id": company_id, "company_name": company_name, "error": "Folder not found"}
            elif total_pages_scraped < 2:
                process_status_id = 4
            else:
                process_status_id = 5

            update_lead_run_history_end(conn, history_id_job_done)
            history_id_processing = insert_lead_run_history(conn, lead_run_id, "Processing results")

            # Step 4: Score results
            save_dir = f"/tmp/RAW_{safe_name}_{max_pages}"  # This is where the scraped pages are stored
            results = kadoa.kadoa_process_results(
                session_id,
                kadoa.kadoa_get_job_results(session_id, kadoa_api_key),
                kadoa_api_key,
                conn,
                lead_run_id,
                save_dir,
                clean=False
            )

            print(results)

            # Step 4.1: Update score
            total_score_value = int(results["total_score"]["Score"].sum())

            update_lead_run_l1_score(conn, lead_run_id, total_score_value)
            update_lead_run_history_end(conn, history_id_processing)

            # Step 4.2: Get documents and upload to S3
            download_company_documents(save_dir, company_name, company_secret, company_url_refresh, env, secrets=secrets)

            # Step 5: Upload to S3
            history_id_upload = insert_lead_run_history(conn, lead_run_id, "Uploading pages to S3")
            base_folder = Path("/tmp") / f"RAW_{safe_name}_{max_pages}"
            uploaded_count, s3_folder_link = kadoa.upload_company_pages_to_s3(
                metadata=results["metadata"],
                company_secret=company_secret,
                env=env,
                base_folder=base_folder,
                scrape_date=datetime.today().strftime("%Y-%m-%d")
            )

            if uploaded_count < 3:
                process_status_id = 4
            else:
                process_status_id = 5

            with conn.cursor() as cursor:
                cursor.execute('''
                    UPDATE "LeadRun"
                    SET "S3FolderLink" = %s
                    WHERE "LeadRunID" = %s
                ''', (s3_folder_link, lead_run_id))
                conn.commit()

            update_lead_run_history_end(conn, history_id_upload)

            # Final update
            update_lead_run_status(conn, lead_run_id, process_status_id)
            history_id_complete = insert_lead_run_history(conn, lead_run_id, "Completed")
            update_lead_run_history_end(conn, history_id_complete)

            # Optional L1+ scoring path (Metrics L1+, SurveyID = 1111)
            # Only run L1+ if scrape was successful (ProcessStatus = 5)
            if scoring_level == 'L1+':
                if process_status_id == 5:
                    try:
                        print(f"Γ£¿ Starting L1+ OpenAI scoring for {company_name}...")

                        # Determine sector to use for CompanyAssessment
                        df_info = rds_get_company_information(conn, company_id=company_id, all_columns=True)
                        if df_info.empty or pd.isna(df_info.get("SectorID", [None])[0]):
                            company_score_sector_id = None
                        else:
                            sector_val = df_info["SectorID"].iloc[0]
                            company_score_sector_id = to_native(sector_val) if sector_val is not None else None

                        # Create CompanyAssessment for L1+ (survey 1111)
                        companyassessment_id = rds_create_company_assessment_id(
                            conn,
                            company_id,
                            survey_id=1111,
                            evaluation_type_id=7,
                            company_score_sector_id=company_score_sector_id,
                        )
                        print(f"Γ£à Created CompanyAssessmentID: {companyassessment_id} for {company_name} (SurveyID=1111)")

                        # Skip PDF generation, use chunking/embedding approach directly with .txt files
                        print(f"≡ƒôÜ Using chunking/embedding approach for L1+ scoring (skipping PDF generation)")

                        # Run L1+ scoring via shared ESG pipeline with chunking
                        print(f"≡ƒñû Generating L1+ score for {company_name} (AssessmentID: {companyassessment_id})")
                        esg.generate_m3_score(
                            conn, 
                            companyassessment_id, 
                            save_dir,  # Pass .txt files folder directly (not PDFs)
                            secrets=secrets,
                            scoring_level='L1+',
                            company_name=company_name,
                            use_chunking=True  # Enable chunking mode
                        )

                        # Update category and topic scores
                        esg.rds_insert_or_update_category_scores(conn, companyassessment_id)
                        esg.rds_insert_or_update_topic_scores(conn, companyassessment_id)

                        # Recompute & write overall CompanyScore (+ sector)
                        print(f"≡ƒº« Recomputing overall company score for AssessmentID {companyassessment_id}")
                        updated_score_df = esg.rds_update_survey_score(conn, companyassessment_id)
                        if updated_score_df is not None and not updated_score_df.empty:
                            print(
                                f"Γ£ö CompanyAssessmentID {int(updated_score_df.iloc[0]['CompanyAssessmentID'])} "
                                f"CompanyScore={float(updated_score_df.iloc[0]['CompanyScore'])} "
                                f"SectorID={int(updated_score_df.iloc[0]['CompanyScoreSectorID'])}"
                            )
                        
                        # Set AssessmentStatusID to 34 (Completed)
                        with conn.cursor() as cursor:
                            cursor.execute('''
                                UPDATE "CompanyAssessment"
                                SET "AssessmentStatusID" = 34, 
                                "SurveyMonkeyModifiedDate" = NOW()
                                WHERE "CompanyAssessmentID" = %s
                            ''', (companyassessment_id,))
                            conn.commit()
                            print(f"Γ£à Set AssessmentStatusID to 34 for CompanyAssessmentID {companyassessment_id}")

                        # Activate the assessment (marks old ones as Inactive)
                        try:
                            print(f"≡ƒöä Activating CompanyAssessmentID {companyassessment_id}...")
                            activation_result = esg.activate_company_assessment(conn, companyassessment_id, rescore=False)
                            if activation_result:
                                deactivated_count = len(activation_result.get('deactivated_assessment_ids', []))
                                if deactivated_count > 0:
                                    print(f"Γ£ö Activated assessment and marked {deactivated_count} previous assessment(s) as Inactive")
                                else:
                                    print("Γ£ö Activated assessment (no previous active assessments found)")
                        except Exception as activation_error:
                            print(f"ΓÜá∩╕Å Failed to activate assessment: {activation_error}")

                    except Exception as l1_plus_error:
                        print(f"ΓÜá∩╕Å L1+ scoring failed for {company_name}: {l1_plus_error}")
                else:
                    print(f"ΓÅ¡∩╕Å Skipping L1+ scoring for {company_name} (scrape not successful, ProcessStatus={process_status_id})")

            elapsed = time.time() - t
            print(f"Total time taken: {elapsed / 60:.2f} minutes")
            print("-" * 100)

            return {
                "company_id": company_id,
                "company_name": company_name,
                "result": results
            }

        except Exception as e:
            conn.rollback()
            if lead_run_id is not None:
                try:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            UPDATE "LeadRun"
                            SET "ProcessStatus" = 3, "EndDate" = %s
                            WHERE "LeadRunID" = %s
                        ''', (datetime.now(), lead_run_id))
                        conn.commit()
                except Exception as update_error:
                    print(f"ΓÜá∩╕Å Failed to update LeadRun with error status: {update_error}")
            else:
                print("ΓÜá∩╕Å lead_run_id not available, skipping DB update for error status")

            print(f"Γ¥î Error processing {company_name} (ID: {company_id}): {e}")

            # If we've already tried once, try again only one more time
            if attempts < max_retries:
                print(f"≡ƒöä Retrying scrape for {company_name} (attempt {attempts+1} of {max_retries})...")
                time.sleep(3)
                continue  # try again
            else:
                print("Γ¥î Max retries reached. Giving up.")
                return {"company_id": company_id, "company_name": company_name, "error": f"Error Processing: {e}"}

def generate_m3_score_single_company(conn, company_id, scoring_level, worker_group, secrets,  uploaded_files=None, company_score_sector_id: int | None = 12, lead_run_id: int | None = None):

    # Validate scoring level
    if scoring_level not in ['L1+', 'L2', 'L3']:
        print(f"Γ¥î Invalid scoring level: {scoring_level}. Must be 'L1+', 'L2', or 'L3'.")
        return {"company_id": company_id, "error": "Invalid scoring level"}

    # Fetch company details
    company_row = None
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        
        if scoring_level == 'L3':
            cursor.execute('''
                SELECT com."CompanyID", "CompanyName" FROM "Company" com
                WHERE com."CompanyID" = %s
            ''', (company_id,))
            company_row = cursor.fetchone()
            
            if not company_row:
                print(f"Γ¥î No Company found with ID: {company_id} for L3 scoring")
                return {"company_id": company_id, "error": "No details found for company"}
            
            company_id   = company_row["CompanyID"]
            company_name = company_row["CompanyName"]
            evaluation_type_id = 6

        else:
            # If lead_run_id is provided, use it directly to find the LeadRun
            # Otherwise, search for LeadRun with ProcessStatus = 5
            if lead_run_id is not None:
                cursor.execute('''
                    SELECT com."CompanyID", "CompanyName", "S3FolderLink", "Secret", "EndDate", lr."LeadRunID" FROM "LeadRun" lr
                    LEFT JOIN "Company" com ON lr."CompanyID" = com."CompanyID"
                    WHERE lr."LeadRunID" = %s AND lr."CompanyID" = %s AND lr."WorkerGroup" = %s
                ''', (lead_run_id, company_id, worker_group))
            else:
                cursor.execute('''
                    SELECT com."CompanyID", "CompanyName", "S3FolderLink", "Secret", "EndDate", lr."LeadRunID" FROM "LeadRun" lr
                    LEFT JOIN "Company" com ON lr."CompanyID" = com."CompanyID"
                    WHERE com."CompanyID" = %s AND lr."WorkerGroup" = %s AND lr."ProcessStatus" = 5
                    ORDER BY lr."EndDate" DESC NULLS LAST
                    LIMIT 1
                ''', (company_id, worker_group))
            company_row = cursor.fetchone()
            
            if not company_row:
                if lead_run_id is not None:
                    print(f"Γ¥î No Company found with ID: {company_id} and LeadRunID: {lead_run_id} for {scoring_level} scoring (WorkerGroup: {worker_group})")
                else:
                    print(f"Γ¥î No Company found with ID: {company_id} for {scoring_level} scoring (WorkerGroup: {worker_group}, ProcessStatus: 5)")
                return {"company_id": company_id, "error": "No details found for company"}
            
            company_id   = company_row["CompanyID"]
            company_name = company_row["CompanyName"]
            s3_folder_link = company_row["S3FolderLink"]
            secret       = company_row["Secret"]
            end_date     = company_row["EndDate"]
            lead_run_id  = company_row["LeadRunID"]
            evaluation_type_id = 7

    base_dir = Path("/tmp")
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', company_name)
    local_path = base_dir / f"{safe_name}_S3_pages"
    cleaned_pdfs_dir = base_dir / f"{safe_name}_cleaned_pdfs"

    try:
        t = time.time()
        print(f"≡ƒÜÇ Starting M3 scoring for {company_name} (ID: {company_id})")

        # Create a CompanyAssessment entry
        companyassessment_id = rds_create_company_assessment_id(
            conn, company_id, survey_id=3333, evaluation_type_id=evaluation_type_id, company_score_sector_id=company_score_sector_id
        )
        print(f"Γ£à Created CompanyAssessmentID: {companyassessment_id} for {company_name}")
        
        # Scoring Processes - L2 and L3
        if scoring_level == 'L2':
            print(f"Starting L2 scoring for {company_name} (ID: {company_id})")
            print(f"S3 Folder Link: {s3_folder_link}")
            print(f"Secret: {secret}")
            print(f"End Date: {end_date}")
            # L2 - Step 1: Update LeadRun with ProcessStatus 6 (Processing)
            update_lead_run_status(conn, lead_run_id, 6)

            # L2 - Step 2: Insert LeadRunHistory for processing
            history_id_processing = insert_lead_run_history(conn, lead_run_id, "Starting L2 scoring process")
            print(f"≡ƒòÆ Inserted LeadRunHistory for processing: {history_id_processing}")

            #L2 - Step 3: download from S3 (skip PDF generation, use chunking)
            if not s3_folder_link:
                return {"error": "No S3 folder link found for L2 scoring"}
            print(f"≡ƒôÑ Downloading pages from S3...")
            esg.download_pages_from_s3(s3_folder_link, local_path, secrets)
            
            # Skip PDF generation, use chunking/embedding approach directly with .txt files
            print(f"≡ƒôÜ Using chunking/embedding approach for L2 scoring (skipping PDF generation)")

            #L2 - Step 4: Score using chunking
            print(f"≡ƒñû Generating L2 score for {company_name} (AssessmentID: {companyassessment_id})")
            esg.generate_m3_score(
                conn, 
                companyassessment_id, 
                local_path,  # Pass .txt files folder directly (not PDFs)
                secrets=secrets,
                scoring_level=scoring_level,
                company_name=company_name,
                use_chunking=True  # Enable chunking mode
            )

            #L2 - Step 5: Update LeadRun with ProcessStatus 8 (Completed)
            update_lead_run_status(conn, lead_run_id, 8)

            #L2 - Step 6: Update LeadRunHistory
            update_lead_run_history_end(conn, history_id_processing)
            history_id_scored = insert_lead_run_history(conn, lead_run_id, "L2 score generated and saved to RDS")
            update_lead_run_history_end(conn, history_id_scored)

            #L2 - Step 7: Insert LeadRunHistory for completion
            history_id_completed = insert_lead_run_history(conn, lead_run_id, "L2 completed")
            update_lead_run_history_end(conn, history_id_completed)

               
        if scoring_level == 'L3' and uploaded_files:
            print(f"Starting L3 scoring for {company_name} (ID: {company_id}) with uploaded files")
            print(f"≡ƒôä Using {len(uploaded_files)} uploaded PDF reports for {company_name}")
            clean_pdfs_path = base_dir / f"{safe_name}_cleaned_pdfs"
            clean_pdfs_path.mkdir(parents=True, exist_ok=True)
            
            # copy the uploaded PDFs into the cleaned folder (no S3 download or text cleaning)
            for file_path in uploaded_files:
                shutil.copy(file_path, clean_pdfs_path)

            esg.generate_m3_score(conn, companyassessment_id, clean_pdfs_path, scoring_level=scoring_level, secrets=secrets)


        # Update CompanyAssessment status
        with conn.cursor() as cursor:
            query = """
                UPDATE public."CompanyAssessment"
                SET "AssessmentStatusID" = 34, "SurveyMonkeyModifiedDate" = NOW()
                WHERE "CompanyAssessmentID" = %s;
            """
            cursor.execute(query, (companyassessment_id,))
            conn.commit()

        # Update Category and Topic Scores
        esg.rds_insert_or_update_category_scores(conn, companyassessment_id)
        esg.rds_insert_or_update_topic_scores(conn, companyassessment_id)

        # Compute & write overall CompanyScore from view.CompanySectorScore (+ sector)
        print(f"≡ƒº« Recomputing overall company score for AssessmentID {companyassessment_id}")
        updated_score_df = esg.rds_update_survey_score(conn, companyassessment_id)
        if updated_score_df is not None and not updated_score_df.empty:
            print(
                f"Γ£ö CompanyAssessmentID {int(updated_score_df.iloc[0]['CompanyAssessmentID'])} "
                f"CompanySectorScore={float(updated_score_df.iloc[0]['CompanySectorScore'])} "
                f"SectorID={int(updated_score_df.iloc[0]['CompanyScoreSectorID'])}"
            )
        else:
            print("ΓÜá∩╕Å No updated overall score returned by rds_update_survey_score")

        # Activate the assessment (marks old ones as Inactive)
        try:
            print(f"≡ƒöä Activating CompanyAssessmentID {companyassessment_id}...")
            activation_result = esg.activate_company_assessment(conn, companyassessment_id, rescore=False)
            if activation_result:
                deactivated_count = len(activation_result.get('deactivated_assessment_ids', []))
                if deactivated_count > 0:
                    print(f"Γ£ö Activated assessment and marked {deactivated_count} previous assessment(s) as Inactive")
                else:
                    print("Γ£ö Activated assessment (no previous active assessments found)")
        except Exception as activation_error:
            print(f"ΓÜá∩╕Å Failed to activate assessment: {activation_error}")

        # Cleanup
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            if os.path.exists(cleaned_pdfs_dir):
                shutil.rmtree(cleaned_pdfs_dir)
        except Exception as cleanup_error:
            print(f"ΓÜá∩╕Å Cleanup error for {company_name}: {cleanup_error}")

        # Return result
        elapsed = time.time() - t
        print(f"M3 {scoring_level} Scoring completed for {company_name} in {elapsed / 60:.2f} minutes")
        print("-" * 100)

        return {
            "company_id": company_id,
            "company_name": company_name,
            "assessment_id": companyassessment_id,
            "updated_company_score": (
                updated_score_df.to_dict(orient="records")[0]
                if updated_score_df is not None and not updated_score_df.empty
                else None
            ),
        }

    except Exception as e:
        conn.rollback()
        if lead_run_id is not None:
            try:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        UPDATE "LeadRun"
                        SET "ProcessStatus" = 7, "EndDate" = %s
                        WHERE "LeadRunID" = %s
                    ''', (datetime.now(), lead_run_id))
                    conn.commit()
            except Exception as update_error:
                print(f"ΓÜá∩╕Å Failed to update LeadRun with error status: {update_error}")

        print(f"Γ¥î Error processing {company_name} (ID: {company_id}): {e}")
        return {
            "company_id": company_id,
            "company_name": company_name,
            "error": str(e)
        }


######## METRICS WIZARD API VALIDATION UTILITIES ########

def validate_company_house_number(company_house_number):
    """
    Validate UK Companies House number format.
    
    Args:
        company_house_number: Company house number string
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    
    Rules:
        - Must be exactly 8 characters
        - Can contain letters and digits (alphanumeric)
        - Examples: "07368089" (8 digits), "NI633481" (2 letters + 6 digits)
    """
    if not company_house_number:
        return False, "Company house number is required"
    
    if not isinstance(company_house_number, str):
        company_house_number = str(company_house_number)
    
    # Remove any whitespace
    company_house_number = company_house_number.strip()
    
    # Check length
    if len(company_house_number) != 8:
        return False, f"Company house number must be exactly 8 characters, got {len(company_house_number)} characters"
    
    # Check if all characters are alphanumeric (letters and digits only)
    if not company_house_number.isalnum():
        return False, "Company house number must contain only letters and digits (a-z, A-Z, 0-9)"
    
    return True, None


def validate_website_url(website):
    """
    Validate and sanitize website URL to prevent injection attacks.
    
    Args:
        website: Website URL string
    
    Returns:
        tuple: (is_valid: bool, sanitized_url: str or None, error_message: str or None)
    
    Rules:
        - Must be a valid URL format
        - Must start with http:// or https://
        - Must not contain dangerous characters or patterns
        - Domain must be valid
    """
    if not website:
        return False, None, "Website URL is required"
    
    if not isinstance(website, str):
        website = str(website)
    
    # Remove any whitespace
    website = website.strip()
    
    # Check for dangerous patterns that could indicate injection attempts
    dangerous_patterns = [
        r'[<>"\']',  # HTML/script injection characters
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data URI
        r'vbscript:',  # VBScript protocol
        r'on\w+\s*=',  # Event handlers (onclick, onerror, etc.)
        r'%00',  # Null byte
        r'\.\./',  # Path traversal
        r'\.\.\\',  # Path traversal (Windows)
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, website, re.IGNORECASE):
            return False, None, f"Website URL contains potentially dangerous characters or patterns"
    
    # Validate URL format
    try:
        parsed = urlparse(website)
        
        # Must have a scheme (http or https)
        if not parsed.scheme:
            # Try adding https:// if no scheme
            if not website.startswith(('http://', 'https://')):
                website = 'https://' + website
                parsed = urlparse(website)
        
        if parsed.scheme not in ['http', 'https']:
            return False, None, "Website URL must use http:// or https:// protocol"
        
        # Must have a netloc (domain)
        if not parsed.netloc:
            return False, None, "Website URL must include a valid domain name"
        
        # Validate domain format (basic check)
        # Domain should contain at least one dot and valid characters
        domain = parsed.netloc.split(':')[0]  # Remove port if present
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$', domain):
            return False, None, "Website URL must contain a valid domain name"
        
        # Reconstruct sanitized URL (scheme + netloc + path, ignore query/fragment for security)
        sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Additional length check to prevent extremely long URLs
        if len(sanitized) > 2048:
            return False, None, "Website URL is too long (maximum 2048 characters)"
        
        return True, sanitized, None
        
    except Exception as e:
        return False, None, f"Invalid website URL format: {str(e)}"


######## METRICS WIZARD API JOB TRACKING ########

import threading
from datetime import datetime, timedelta

def get_assessment_job_status(conn, company_id, worker_group=None, lead_run_id=None):
    """
    Get assessment job status from existing database tables (LeadRun and CompanyAssessment).
    This replaces UUID-based job tracking by querying the actual workflow state.
    
    Args:
        conn: Database connection
        company_id: Company ID
        worker_group: Worker group name (optional). If lead_run_id is provided, this is ignored.
        lead_run_id: LeadRun ID (optional). If provided, returns status for that specific LeadRun.
                     Takes precedence over worker_group.
    
    Returns:
        dict or None: Job status with keys:
            - status: "queued" | "running" | "scraping" | "scoring" | "completed" | "failed"
                      Maps from ProcessStatus table values (Category - Description):
                      1="Scrape" - "Started Scrape",
                      2="Scrape" - "Scrape Finished, Started Processing",
                      3="Scrape" - "Error Processing",
                      4="Scrape" - "Blocked website less 3 pages",
                      5="Scrape" - "Complete",
                      6="L2 AI" - "Started Processing",
                      7="L2 AI" - "Error Processing",
                      8="L2 AI" - "Complete",
                      9="Scrape" - "To do"
            - progress: 0-100
            - message: Status message
            - error_message: Error message if failed (optional)
            - result: dict with company_id and assessment_id if completed (optional)
            - lead_run_id: LeadRunID if available (optional)
            - assessment_id: CompanyAssessmentID if available (optional)
        Returns None if no assessment job exists for the company (job never triggered)
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Get LeadRun - lead_run_id takes precedence over worker_group
            if lead_run_id:
                cursor.execute('''
                    SELECT 
                        "LeadRunID",
                        "ProcessStatus",
                        "StartDate",
                        "EndDate",
                        "MainURL",
                        "S3FolderLink"
                    FROM "LeadRun"
                    WHERE "LeadRunID" = %s
                      AND "CompanyID" = %s
                ''', (lead_run_id, company_id))
            elif worker_group:
                cursor.execute('''
                    SELECT 
                        "LeadRunID",
                        "ProcessStatus",
                        "StartDate",
                        "EndDate",
                        "MainURL",
                        "S3FolderLink"
                    FROM "LeadRun"
                    WHERE "CompanyID" = %s 
                      AND "WorkerGroup" = %s
                    ORDER BY "StartDate" DESC
                    LIMIT 1
                ''', (company_id, worker_group))
            else:
                cursor.execute('''
                    SELECT 
                        "LeadRunID",
                        "ProcessStatus",
                        "StartDate",
                        "EndDate",
                        "MainURL",
                        "S3FolderLink"
                    FROM "LeadRun"
                    WHERE "CompanyID" = %s
                    ORDER BY "StartDate" DESC
                    LIMIT 1
                ''', (company_id,))
            
            lead_run = cursor.fetchone()
            
            # Get most recent LeadRunHistory for progress message
            latest_history = None
            if lead_run:
                cursor.execute('''
                    SELECT "Process", "StartDate", "EndDate"
                    FROM "LeadRunHistory"
                    WHERE "LeadRunID" = %s
                    ORDER BY "StartDate" DESC
                    LIMIT 1
                ''', (lead_run["LeadRunID"],))
                latest_history = cursor.fetchone()
            
            # Get most recent CompanyAssessment for L2 scoring (SurveyID = 3333)
            cursor.execute('''
                SELECT 
                    "CompanyAssessmentID",
                    "AssessmentStatusID",
                    "AssessmentValidStatusID",
                    "CreateDate",
                    "UpdateDate",
                    "CompanyScore"
                FROM "CompanyAssessment"
                WHERE "CompanyID" = %s
                  AND "SurveyID" = 3333
                ORDER BY "CreateDate" DESC
                LIMIT 1
            ''', (company_id,))
            
            assessment = cursor.fetchone()
            
            # Determine status based on LeadRun ProcessStatus and CompanyAssessment
            if not lead_run:
                # No LeadRun exists - no job has been triggered for this company
                return None
            
            process_status = lead_run["ProcessStatus"]
            lead_run_id = lead_run["LeadRunID"]
            
            # Map ProcessStatus to job status
            # ProcessStatus values (from ProcessStatus table):
            # 1 = "Scrape" - "Started Scrape"
            # 2 = "Scrape" - "Scrape Finished, Started Processing"
            # 3 = "Scrape" - "Error Processing"
            # 4 = "Scrape" - "Blocked website less 3 pages"
            # 5 = "Scrape" - "Complete"
            # 6 = "L2 AI" - "Started Processing"
            # 7 = "L2 AI" - "Error Processing"
            # 8 = "L2 AI" - "Complete"
            # 9 = "Scrape" - "To do" (initial/queued state)
            
            if process_status in [3, 4, 7]:
                # Failed states
                error_messages = {
                    3: "Error processing during scraping",
                    4: "Website blocked or insufficient pages (< 3 pages)",
                    7: "Error processing during L2 AI scoring"
                }
                return {
                    "status": "failed",
                    "progress": 0,
                    "message": error_messages.get(process_status, "Assessment failed"),
                    "error_message": error_messages.get(process_status, "Unknown error"),
                    "company_id": company_id,
                    "lead_run_id": lead_run_id
                }
            
            elif process_status == 9:
                # Initial/queued state
                return {
                    "status": "queued",
                    "progress": 0,
                    "message": "Assessment queued - waiting to start",
                    "company_id": company_id,
                    "lead_run_id": lead_run_id
                }
            
            elif process_status in [1, 2]:
                # Scraping in progress
                progress_msg = latest_history["Process"] if latest_history else "Scraping website..."
                progress_pct = 25 if process_status == 1 else 40
                return {
                    "status": "scraping",
                    "progress": progress_pct,
                    "message": progress_msg,
                    "company_id": company_id,
                    "lead_run_id": lead_run_id
                }
            
            elif process_status == 5:
                # Scraping completed, check if scoring has started
                if assessment:
                    assessment_status = assessment["AssessmentStatusID"]
                    if assessment_status == 34 and assessment.get("AssessmentValidStatusID") == 2:
                        # Assessment completed
                        return {
                            "status": "completed",
                            "progress": 100,
                            "message": "Assessment completed successfully",
                            "company_id": company_id,
                            "lead_run_id": lead_run_id,
                            "assessment_id": assessment["CompanyAssessmentID"],
                            "result": {
                                "company_id": company_id,
                                "assessment_id": assessment["CompanyAssessmentID"]
                            }
                        }
                    elif assessment_status == 18:
                        # Scoring in progress
                        progress_msg = latest_history["Process"] if latest_history else "Generating scores..."
                        return {
                            "status": "scoring",
                            "progress": 75,
                            "message": progress_msg,
                            "company_id": company_id,
                            "lead_run_id": lead_run_id,
                            "assessment_id": assessment["CompanyAssessmentID"]
                        }
                else:
                    # Scraping done, scoring not started yet
                    return {
                        "status": "scraping",
                        "progress": 50,
                        "message": "Scraping completed - preparing for scoring",
                        "company_id": company_id,
                        "lead_run_id": lead_run_id
                    }
            
            elif process_status == 6:
                # L2 AI scoring in progress
                progress_msg = latest_history["Process"] if latest_history else "Generating L2 scores..."
                assessment_id = assessment["CompanyAssessmentID"] if assessment else None
                return {
                    "status": "scoring",
                    "progress": 75,
                    "message": progress_msg,
                    "company_id": company_id,
                    "lead_run_id": lead_run_id,
                    "assessment_id": assessment_id
                }
            
            elif process_status == 8:
                # L2 AI scoring completed
                if assessment and assessment["AssessmentStatusID"] == 34:
                    return {
                        "status": "completed",
                        "progress": 100,
                        "message": "Assessment completed successfully",
                        "company_id": company_id,
                        "lead_run_id": lead_run_id,
                        "assessment_id": assessment["CompanyAssessmentID"],
                        "result": {
                            "company_id": company_id,
                            "assessment_id": assessment["CompanyAssessmentID"]
                        }
                    }
                else:
                    return {
                        "status": "scoring",
                        "progress": 90,
                        "message": "L2 scoring completed - finalizing assessment",
                        "company_id": company_id,
                        "lead_run_id": lead_run_id,
                        "assessment_id": assessment["CompanyAssessmentID"] if assessment else None
                    }
            
            else:
                # Unknown status - default to running
                progress_msg = latest_history["Process"] if latest_history else "Processing..."
                return {
                    "status": "running",
                    "progress": 25,
                    "message": progress_msg,
                    "company_id": company_id,
                    "lead_run_id": lead_run_id
                }
    
    except Exception as e:
        logging.exception(f"Error getting assessment job status for company {company_id}")
        return {
            "status": "failed",
            "progress": 0,
            "message": "Error retrieving job status",
            "error_message": str(e),
            "company_id": company_id
        }


######## METRICS WIZARD API TRANSFORMATION FUNCTIONS ########

def map_category_name(db_category_name):
    """
    Map database category name to frontend category enum.
    
    Args:
        db_category_name: Category description from database (e.g., "Environmental", "Social", "Governance")
    
    Returns:
        str: One of "Environmental", "Social", or "Governance"
    """
    if not db_category_name:
        return "Environmental"  # Default
    
    category_map = {
        "environmental": "Environmental",
        "social": "Social",
        "governance": "Governance"
    }
    
    return category_map.get(db_category_name.lower(), "Environmental")


def transform_company_lookup(company_row, assessment_row, website_error_row):
    """
    Transform database results to frontend MockCompanyRecord format.
    
    Args:
        company_row: Dictionary with company data from database query
        assessment_row: Dictionary with recent assessment data (or None)
        website_error_row: Dictionary with website error data (or None)
    
    Returns:
        dict: Formatted response matching MockCompanyRecord interface
    """
    if not company_row:
        return {
            "exists": False,
            "websiteBlocked": False
        }
    
    # Check if website was blocked
    website_blocked = website_error_row is not None
    
    # Check if employee/revenue data is missing
    has_missing_data = (
        not company_row.get("number_of_employee_id") or 
        not company_row.get("metric_revenue_id") or
        not company_row.get("employee_range") or
        not company_row.get("revenue_range")
    )
    
    # Format assessment date and determine recency (within last 13 months)
    assessment_date = None
    has_recent_assessment = False
    if assessment_row and assessment_row.get("assessmentDate"):
        raw_date = assessment_row["assessmentDate"]
        if isinstance(raw_date, datetime):
            # Return date-only (YYYY-MM-DD) to the frontend
            try:
                assessment_date = raw_date.date().isoformat()
            except Exception:
                assessment_date = raw_date.isoformat()
            try:
                # Consider assessment "recent" if within approximately the last 13 months
                from datetime import timedelta
                now = datetime.utcnow()
                # 13 months Γëê 395 days; adjust if you need exact calendar months
                if raw_date >= now - timedelta(days=395):
                    has_recent_assessment = True
            except Exception:
                # Fallback: if we can't safely compare, treat as non-recent but still return the date
                has_recent_assessment = False
        else:
            # For non-datetime types, just coerce to string
            assessment_date = str(raw_date)
    
    return {
        "exists": True,
        "websiteBlocked": website_blocked,
        "companyData": {
            "name": company_row.get("CompanyName", ""),
            "sector": company_row.get("sector") or "Unknown",
            "industry": company_row.get("industry") or "Unknown",
            "sectorId": company_row.get("SectorID"),
            "industryId": company_row.get("IndustryID"),
            "companyId": company_row.get("CompanyID"),
            "employee_range": company_row.get("employee_range") or "",
            "number_of_employee_id": company_row.get("number_of_employee_id"),
            "revenue_range": company_row.get("revenue_range") or "",
            "metric_revenue_id": company_row.get("metric_revenue_id"),
            "hasRecentAssessment": has_recent_assessment,
            "assessmentDate": assessment_date
        }
    }


def fetch_company_for_lookup(conn, company_house_number, website):
    """
    Fetch company data for the Metrics Wizard lookup endpoint.

    This encapsulates the SQL used to join Company, Sector, Industry,
    NumberOfEmployee, MetricRevenue, and CompanyAssessment to produce
    a single row used by transform_company_lookup.

    Args:
        conn: psycopg2 connection
        company_house_number (str): Companies House number
        website (str): Sanitized website URL

    Returns:
        dict | None: Row dict if found, else None.
    """
    with conn.cursor() as cursor:
        query = """
            SELECT 
                c."CompanyID",
                c."CompanyName",
                c."CompanyWebSite",
                c."CompanyHouseNumber",
                c."SectorID",
                s."SectorDescription" as sector,
                c."IndustryID",
                i."IndustryDescription" as industry,
                c."NumberOfEmployeeID" as "number_of_employee_id",
                MAX(COALESCE(noe."NumberOfEmployeeDescription", '')) as "employee_range",
                c."MetricRevenueID" as "metric_revenue_id",
                MAX(COALESCE(mr."MetricRevenueDescription", '')) as "revenue_range",
                MAX(ca."CreateDate") AS latest_assessment_date
            FROM "Company" c
            LEFT JOIN "Sector" s ON c."SectorID" = s."SectorID"
            LEFT JOIN "Industry" i ON c."IndustryID" = i."IndustryID"
            LEFT JOIN "NumberOfEmployee" noe ON c."NumberOfEmployeeID" = noe."NumberOfEmployeeID"
            LEFT JOIN "MetricRevenue" mr ON c."MetricRevenueID" = mr."MetricRevenueID"
            LEFT JOIN "CompanyAssessment" ca 
                   ON ca."CompanyID" = c."CompanyID"
                  AND ca."SurveyID" = 3333
                  AND ca."AssessmentValidStatusID" = 2
            WHERE c."CompanyHouseNumber" = %s
              AND REGEXP_REPLACE(LOWER(c."CompanyWebSite"), '^(https?://)?(www\\.)?', '') = 
                  REGEXP_REPLACE(LOWER(%s), '^(https?://)?(www\\.)?', '')
            GROUP BY 
                c."CompanyID",
                c."CompanyName",
                c."CompanyWebSite",
                c."CompanyHouseNumber",
                c."SectorID",
                s."SectorDescription",
                c."IndustryID",
                i."IndustryDescription",
                c."NumberOfEmployeeID",
                c."MetricRevenueID"
            ORDER BY latest_assessment_date DESC NULLS LAST,
                     c."CompanyID" ASC
            LIMIT 1;
        """
        cursor.execute(query, (company_house_number, website))
        row = cursor.fetchone()
        if not row:
            return None

        cols = [desc[0] for desc in cursor.description]
        company_row = dict(zip(cols, row))

        logging.info(
            "Company lookup result - CompanyID: %s, number_of_employee_id: %s, "
            "employee_range: '%s', metric_revenue_id: %s, revenue_range: '%s'",
            company_row.get("CompanyID"),
            company_row.get("number_of_employee_id"),
            company_row.get("employee_range"),
            company_row.get("metric_revenue_id"),
            company_row.get("revenue_range"),
        )

        return company_row


def fetch_website_error_for_company(conn, company_id):
    """
    Fetch latest website error (LeadRun ProcessStatus=9) for a company.

    Args:
        conn: psycopg2 connection
        company_id (int): Company ID

    Returns:
        dict | None: Row dict if a website error exists, else None.
    """
    with conn.cursor() as cursor:
        query = """
            SELECT 
                lr."LeadRunID",
                lr."ProcessStatus",
                lr."CompanyID"
            FROM "LeadRun" lr
            WHERE lr."CompanyID" = %s
              AND lr."ProcessStatus" = 9
            ORDER BY lr."EndDate" DESC NULLS LAST,
                     lr."LeadRunID" DESC
            LIMIT 1;
        """
        cursor.execute(query, (company_id,))
        row = cursor.fetchone()
        if not row:
            return None

        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))


def fetch_company_details(conn, company_id):
    """
    Fetch company data by company_id for the Metrics Wizard company-details endpoint.
    
    Args:
        conn: psycopg2 connection
        company_id (int): Company ID
    
    Returns:
        dict | None: Row dict if found, else None.
    """
    with conn.cursor() as cursor:
        query = """
            SELECT 
                c."CompanyID",
                c."CompanyName",
                c."CompanyWebSite",
                c."CompanyHouseNumber",
                c."SectorID",
                s."SectorDescription" as sector,
                c."IndustryID",
                i."IndustryDescription" as industry,
                c."NumberOfEmployeeID" as "number_of_employee_id",
                COALESCE(noe."NumberOfEmployeeDescription", '') as "employee_range",
                c."MetricRevenueID" as "metric_revenue_id",
                COALESCE(mr."MetricRevenueDescription", '') as "revenue_range",
                MAX(ca."CreateDate") AS latest_assessment_date
            FROM "Company" c
            LEFT JOIN "Sector" s ON c."SectorID" = s."SectorID"
            LEFT JOIN "Industry" i ON c."IndustryID" = i."IndustryID"
            LEFT JOIN "NumberOfEmployee" noe ON c."NumberOfEmployeeID" = noe."NumberOfEmployeeID"
            LEFT JOIN "MetricRevenue" mr ON c."MetricRevenueID" = mr."MetricRevenueID"
            LEFT JOIN "CompanyAssessment" ca 
                   ON ca."CompanyID" = c."CompanyID"
                  AND ca."SurveyID" = 3333
                  AND ca."AssessmentValidStatusID" = 2
            WHERE c."CompanyID" = %s
            GROUP BY 
                c."CompanyID",
                c."CompanyName",
                c."CompanyWebSite",
                c."CompanyHouseNumber",
                c."SectorID",
                s."SectorDescription",
                c."IndustryID",
                i."IndustryDescription",
                c."NumberOfEmployeeID",
                noe."NumberOfEmployeeDescription",
                c."MetricRevenueID",
                mr."MetricRevenueDescription"
            LIMIT 1;
        """
        cursor.execute(query, (company_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))


def transform_company_details(company_row, website_error_row, l2_data):
    """
    Transform database results to frontend CompanyData format with optional l2Data.
    
    Args:
        company_row: Dictionary with company data from database query
        website_error_row: Dictionary with website error data (or None)
        l2_data: Dictionary with L2 assessment data (or None)
    
    Returns:
        dict: Formatted response matching CompanyData interface
    """
    if not company_row:
        return None  # Will result in 404 in endpoint
    
    # Check if website was blocked
    website_blocked = website_error_row is not None
    
    # Format assessment date and determine recency (within last 13 months = 395 days)
    assessment_date = None
    has_recent_assessment = False
    latest_assessment_date = company_row.get("latest_assessment_date")
    
    if latest_assessment_date:
        if isinstance(latest_assessment_date, datetime):
            # Return date-only (YYYY-MM-DD) to the frontend
            try:
                assessment_date = latest_assessment_date.date().isoformat()
            except Exception:
                assessment_date = latest_assessment_date.isoformat()
            try:
                from datetime import timedelta
                now = datetime.utcnow()
                if latest_assessment_date >= now - timedelta(days=395):
                    has_recent_assessment = True
            except Exception:
                has_recent_assessment = False
        else:
            assessment_date = str(latest_assessment_date)
    
    # Build CompanyData structure
    result = {
        "companyId": company_row.get("CompanyID"),
        "name": company_row.get("CompanyName", ""),
        "sector": company_row.get("sector") or "Unknown",
        "sectorId": company_row.get("SectorID"),
        "industry": company_row.get("industry") or "Unknown",
        "industryId": company_row.get("IndustryID"),
        "employee_range": company_row.get("employee_range") or "",
        "number_of_employee_id": company_row.get("number_of_employee_id"),
        "revenue_range": company_row.get("revenue_range") or "",
        "metric_revenue_id": company_row.get("metric_revenue_id"),
        "hasRecentAssessment": has_recent_assessment,
        "assessmentDate": assessment_date,
        "websiteBlocked": website_blocked,
        "l2Data": l2_data  # Will be None if no assessment
    }
    
    return result


def transform_benchmark_data(
    sector_stats,
    industry_stats,
    revenue_ranges,
    employee_ranges,
    word_cloud_terms,
    company_revenue_range,
    company_employee_range,
    decimal_places: int = 0,
):
    """
    Transform benchmark data to frontend BenchmarkData format.
    
    Args:
        sector_stats: Dict with sector average, min, max, label
        industry_stats: Dict with industry average, min, max, label
        revenue_ranges: List of dicts with range, minScore, maxScore
        employee_ranges: List of dicts with range, minScore, maxScore
        word_cloud_terms: List of dicts with text and count/weight
        company_revenue_range: String of company's revenue range
        company_employee_range: String of company's employee range
    
    Returns:
        dict: Formatted response matching BenchmarkData interface
    """
    def _round(value):
        if value is None:
            return None
        try:
            return round(float(value), decimal_places) if decimal_places is not None else float(value)
        except (TypeError, ValueError):
            return value

    # Calculate sector score dial
    sector_avg = float(sector_stats.get("sectorAverage", 0)) if sector_stats.get("sectorAverage") is not None else 0
    sector_min = float(sector_stats.get("sectorMin", 0)) if sector_stats.get("sectorMin") is not None else 0
    sector_max = float(sector_stats.get("sectorMax", 100)) if sector_stats.get("sectorMax") is not None else 100
    sector_score = {
        "value": _round(sector_avg),
        "min": _round(sector_min),
        "max": _round(sector_max),
        "label": sector_stats.get("sectorLabel", "Sector Average")
    }
    
    # Calculate industry score dial
    if industry_stats and industry_stats.get("industryAverage") is not None:
        industry_avg = float(industry_stats.get("industryAverage", 0))
        industry_min = float(industry_stats.get("industryMin", 0)) if industry_stats.get("industryMin") is not None else 0
        industry_max = float(industry_stats.get("industryMax", 100)) if industry_stats.get("industryMax") is not None else 100
        industry_label = industry_stats.get("industryLabel", "Industry Average")
    else:
        # No industry specified or no data: return nulls so frontend can treat as missing
        industry_avg = None
        industry_min = None
        industry_max = None
        industry_label = None

    industry_score = {
        "value": _round(industry_avg) if industry_avg is not None else None,
        "min": _round(industry_min) if industry_min is not None else None,
        "max": _round(industry_max) if industry_max is not None else None,
        "label": industry_label,
    }
    
    # Transform revenue ranges
    revenue_range_scores = []
    for row in revenue_ranges:
        revenue_range_scores.append({
            "range": row.get("revenueRange", ""),
            "minScore": _round(row.get("minScore", 0)),
            "maxScore": _round(row.get("maxScore", 100)),
            "isCompanyRange": row.get("revenueRange") == company_revenue_range
        })
    
    # Transform employee ranges
    employee_range_scores = []
    for row in employee_ranges:
        employee_range_scores.append({
            "range": row.get("employeeRange", ""),
            "minScore": _round(row.get("minScore", 0)),
            "maxScore": _round(row.get("maxScore", 100)),
            "isCompanyRange": row.get("employeeRange") == company_employee_range
        })
    
    # Transform word cloud (normalize weights to 1-100 scale)
    word_cloud_data = []
    if word_cloud_terms:
        max_count = max([term.get("totalCount", 0) or term.get("weight", 0) for term in word_cloud_terms]) or 1
        for term in word_cloud_terms:
            count = term.get("totalCount", 0) or term.get("weight", 0)
            weight = int((count / max_count) * 100) if max_count > 0 else 0
            word_cloud_data.append({
                "text": term.get("text", ""),
                "weight": weight
            })
    
    return {
        "sectorScore": sector_score,
        "industryScore": industry_score,
        "revenueRanges": revenue_range_scores,
        "employeeRanges": employee_range_scores,
        "wordCloudData": word_cloud_data
    }


def transform_assessment_to_l2_data(assessment_row, topic_scores, sector_stats, topic_benchmarks):
    """
    Transform assessment data to frontend L2Data format.
    
    Args:
        assessment_row: Dict with assessment data (companyScore, maxScore, etc.)
        topic_scores: List of dicts with topic score data from view_assessmenttopicscore
        sector_stats: Dict with sector benchmark stats (from rds_get_sector_benchmark_stats)
        topic_benchmarks: Dict with benchmark data by topic_id: {sector: {}, industry: {}, revenue: {}}
    
    Returns:
        dict: Formatted response matching L2Data interface
    """
    # Group topics by category
    environmental_topics = []
    social_topics = []
    governance_topics = []
    
    all_topics_with_scores = []
    
    for topic_row in topic_scores:
        category_name = topic_row.get("CategoryDescription", "")
        topic_name = topic_row.get("TopicDescription", "")
        topic_id = topic_row.get("TopicID")
        
        # Get benchmark averages for this topic and round to 0 decimals
        sector_avg = 0
        industry_avg = float(topic_row.get("MeanTopicScore", 0)) if topic_row.get("MeanTopicScore") else 0
        revenue_avg = 0
        
        if topic_benchmarks and topic_id:
            if "sector" in topic_benchmarks and topic_id in topic_benchmarks["sector"]:
                sector_avg = float(topic_benchmarks["sector"][topic_id].get("sectorAverage", 0)) or 0
            if "revenue" in topic_benchmarks and topic_id in topic_benchmarks["revenue"]:
                revenue_avg = float(topic_benchmarks["revenue"][topic_id].get("revenueAverage", 0)) or 0
        
        # Get and round topic score and all averages to 0 decimals
        topic_score_raw = float(topic_row.get("TopicScore", 0)) if topic_row.get("TopicScore") else 0
        topic_score_rounded = round(topic_score_raw)
        sector_avg = round(sector_avg)
        industry_avg = round(industry_avg)
        revenue_avg = round(revenue_avg)
        
        # Create TopicScore for spider chart (radar plot)
        topic_score = {
            "topic": topic_name,
            "companyScore": topic_score_rounded,
            "industryAverage": industry_avg
        }
        
        # Create ScoringTopic object for table
        scoring_topic = {
            "name": topic_name,
            "description": topic_row.get("TopicExplanation", "") or "",
            "score": topic_score_rounded,
            "sectorAverage": sector_avg,
            "industryAverage": industry_avg,
            "revenueAverage": revenue_avg,
            "category": map_category_name(category_name)
        }
        
        all_topics_with_scores.append(scoring_topic)
        
        # Add to category-specific arrays
        category_lower = category_name.lower()
        if "environmental" in category_lower:
            environmental_topics.append(topic_score)
        elif "social" in category_lower:
            social_topics.append(topic_score)
        elif "governance" in category_lower:
            governance_topics.append(topic_score)
    
    # Sort all topics by score and get top 5 / bottom 5
    sorted_topics = sorted(all_topics_with_scores, key=lambda x: x["score"], reverse=True)
    top5_topics = sorted_topics[:5]
    bottom5_topics = sorted_topics[-5:] if len(sorted_topics) >= 5 else sorted_topics
    
    # Calculate maxScore (typically 300 for Metrics 3, but should be configurable)
    max_score = float(assessment_row.get("maxScore", 300)) if assessment_row.get("maxScore") else 300
    
    # Get sector average for total score (from sector_stats, same as benchmark endpoint)
    sector_avg_total = round(float(sector_stats.get("sectorAverage", 0)) if sector_stats.get("sectorAverage") else 0)
    
    # Get companyScore from assessment_row, but if it's 0 or missing, calculate from topic scores
    company_score = float(assessment_row.get("companyScore", 0)) if assessment_row.get("companyScore") else 0
    if company_score == 0 and all_topics_with_scores:
        # Fallback: calculate companyScore as sum of all topic scores
        company_score = sum(topic["score"] for topic in all_topics_with_scores)
    company_score = round(company_score)
    
    # Restructure into better nested format
    return {
        "companyScore": {
            "score": company_score,
            "sectorAverage": sector_avg_total
        },
        "maxScore": max_score,
        "topicScoresByCategory": {
            "environmental": environmental_topics,
            "social": social_topics,
            "governance": governance_topics
        },
        "topTopics": top5_topics,
        "bottomTopics": bottom5_topics
    }


def fetch_l2_data_for_company(conn, company_id):
    """
    Fetch full L2 assessment data for a company.

    This encapsulates the logic previously implemented in the
    /api/v1/metrics-wizard/assessment-results/<company_id> endpoint so it can
    be reused by multiple endpoints.

    Args:
        conn: psycopg2 connection
        company_id (int): Company ID

    Returns:
        dict | None: L2Data-style dict if an active assessment exists,
        else None.
    """
    # Step 1: Get most recent active assessment and company score from view
    with conn.cursor() as cursor:
        query = """
            SELECT 
                ca."CompanyAssessmentID",
                ca."CreateDate",
                c."IndustryID",
                c."SectorID",
                c."MetricRevenueID",
                c."NumberOfEmployeeID"
            FROM "CompanyAssessment" ca
            INNER JOIN "Company" c ON ca."CompanyID" = c."CompanyID"
            WHERE ca."CompanyID" = %s
              AND ca."SurveyID" = 3333
              AND ca."AssessmentValidStatusID" = 2
            ORDER BY ca."CreateDate" DESC
            LIMIT 1;
        """
        cursor.execute(query, (company_id,))
        row = cursor.fetchone()
        if not row:
            return None

        cols = [desc[0] for desc in cursor.description]
        assessment_row = dict(zip(cols, row))
        company_assessment_id = assessment_row.get("CompanyAssessmentID")
        industry_id = assessment_row.get("IndustryID")
        sector_id = assessment_row.get("SectorID")
        metric_revenue_id = assessment_row.get("MetricRevenueID")
        
        # Get company score from view_assessment_totalscore
        query_total = """
            SELECT "TotalScore"
            FROM view_assessment_totalscore
            WHERE "CompanyAssessmentID" = %s;
        """
        cursor.execute(query_total, (company_assessment_id,))
        total_row = cursor.fetchone()
        if total_row:
            assessment_row["companyScore"] = round(float(total_row[0]) if total_row[0] is not None else 0)
        else:
            assessment_row["companyScore"] = 0

    # Step 2: Get topic scores using view_assessmenttopicscore
    topic_scores = []
    with conn.cursor() as cursor:
        query = """
            SELECT 
                ats."CategoryID",
                cat."CategoryDescription",
                ats."TopicID",
                ats."TopicDescription",
                ats."TopicScore",
                ats."MeanTopicScore",
                t."TopicExplanation"
            FROM "view_assessmenttopicscore" ats
            INNER JOIN "Category" cat ON ats."CategoryID" = cat."CategoryID"
            INNER JOIN "Topic" t ON ats."TopicID" = t."TopicID"
            WHERE ats."CompanyAssessmentID" = %s
              AND ats."SurveyID" = 3333
            ORDER BY cat."CategoryReportSequence", t."TopicReportSequence";
        """
        cursor.execute(query, (company_assessment_id,))
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        for row in rows:
            topic_scores.append(dict(zip(cols, row)))

    # Step 3: Get sector average for total score (using same method as benchmark endpoint)
    # Note: This actually calculates sector average, not industry average, despite variable name
    sector_stats = {}
    if sector_id:
        sector_stats = rds_get_sector_benchmark_stats(conn, sector_id)
    
    # Also get industry average if industry_id is available
    industry_stats = {}
    if industry_id:
        industry_stats = rds_get_industry_benchmark_stats(conn, industry_id)

    # Step 4: Get benchmark averages for topics
    topic_benchmarks = {"sector": {}, "industry": {}, "revenue": {}}

    # Get sector averages for topics
    if sector_id:
        with conn.cursor() as cursor:
            query = """
                SELECT 
                    bmt."TopicID",
                    AVG(bmt."Mean") as sectorAverage
                FROM "view_benchmarkmetrictopic" bmt
                WHERE bmt."SurveyID" = 3333
                  AND bmt."SectorID" = %s
                  AND bmt."Snap Date" >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY bmt."TopicID";
            """
            cursor.execute(query, (sector_id,))
            rows = cursor.fetchall()
            for row in rows:
                topic_id = row[0]
                sector_avg = float(row[1]) if row[1] else 0
                topic_benchmarks["sector"][topic_id] = {"sectorAverage": sector_avg}

    # Get revenue range averages for topics
    if metric_revenue_id:
        with conn.cursor() as cursor:
            query = """
                SELECT 
                    bmt."TopicID",
                    AVG(bmt."Mean") as revenueAverage
                FROM "view_benchmarkmetrictopic" bmt
                WHERE bmt."SurveyID" = 3333
                  AND bmt."SectorID" = %s
                  AND bmt."Metric Revenue" = (
                      SELECT "MetricRevenueDescription" 
                      FROM "MetricRevenue" 
                      WHERE "MetricRevenueID" = %s
                  )
                  AND bmt."Snap Date" >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY bmt."TopicID";
            """
            cursor.execute(query, (sector_id, metric_revenue_id))
            rows = cursor.fetchall()
            for row in rows:
                topic_id = row[0]
                revenue_avg = float(row[1]) if row[1] else 0
                topic_benchmarks["revenue"][topic_id] = {"revenueAverage": revenue_avg}

    # Set maxScore (typically 300 for Metrics 3)
    assessment_row["maxScore"] = 300

    # Transform to frontend format
    # Pass sector_stats as second parameter (it will extract sectorAverage from it)
    return transform_assessment_to_l2_data(
        assessment_row, topic_scores, sector_stats, topic_benchmarks
    )


# =============================================================================
# Metrics Wizard - Companies House Integration Functions
# =============================================================================

def rds_get_companyhouse_data(conn, company_house_number):
    """
    Retrieve company information from view_companyhouse with SIC code mapping.
    
    Args:
        conn: psycopg2 connection
        company_house_number (str): UK Companies House registration number (8 alphanumeric chars)
    
    Returns:
        dict | None: Company house data with mapped sector/industry, or None if not found
    """
    with conn.cursor() as cursor:
        query = """
            SELECT
                ch."CompanyHouseID",
                ch."CompanyName",
                ch."CompanyNumber",
                ch."CareOf",
                ch."POBox",
                ch."AddressLine1",
                ch."AddressLine2",
                ch."PostTown",
                ch."County",
                ch."Country",
                ch."PostCodeID",
                ch."PostCode",
                ch."Latitude",
                ch."Longitude",
                ch."CompanyHouseCategoryID",
                ch."CompanyHouseCategoryDescription",
                ch."CompanyHouseStatusID",
                ch."CompanyHouseStatusDescription",
                ch."DissolutionDate",
                ch."IncorporationDate",
                ch."CountryOfOrigin",
                ch."CompanyHouseAccountCategoryID",
                ch."CompanyHouseAccountCategoryDescription",
                ch."SicText_1",
                ch."SicText_2",
                ch."SicText_3",
                ch."SicText_4",
                ch."SIC_Description",
                ch."SectionID",
                ch."SectionDescription",
                ch."DivisionID",
                ch."DivisionDescription",
                ch."GroupID",
                ch."GroupDescription",
                ch."ClassID",
                ch."ClassDescription",
                ch."Class_SubID",
                ch."Class_SubDescription",
                ch."URI",
                si."SIC_CodeID",
                i."IndustryID",
                i."IndustryDescription",
                i."IndustryDetail",
                s."SectorID",
                s."SectorDescription",
                s."ESGTheme"
            FROM public.view_companyhouse ch
            LEFT JOIN public."SIC_Industry" si
                ON si."SIC_CodeID" = ch."SicText_1"
            LEFT JOIN public."Industry" i
                ON i."IndustryID" = si."IndustryID"
            LEFT JOIN public."Sector" s
                ON s."SectorID" = i."SectorID"
            WHERE ch."CompanyNumber" = %s
        """
        cursor.execute(query, (company_house_number,))
        
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Convert to dictionary
        result = dict(zip(columns, row))
        
        logging.info(
            f"Retrieved Companies House data for {company_house_number}: "
            f"Name='{result.get('CompanyName')}', SIC={result.get('SicText_1')}, "
            f"Sector={result.get('SectorDescription')}"
        )
        
        return result


def determine_mapping_status(sic_code, sector_id, industry_id):
    """
    Determine the SIC code to sector/industry mapping status.
    
    Args:
        sic_code (str | None): Primary SIC code
        sector_id (int | None): Mapped sector ID
        industry_id (int | None): Mapped industry ID
    
    Returns:
        tuple: (status, message) where status is one of:
            - "success": SIC code successfully mapped
            - "unmapped_sic_code": SIC code exists but not mapped
            - "no_sic_code": No SIC code available
    """
    if sic_code is None or sic_code == '':
        return (
            "no_sic_code",
            "No SIC code available for this company. Please select sector/industry manually."
        )
    
    if sector_id is None:
        # Log unmapped SIC code for data remediation
        logging.warning(
            f"Unmapped SIC code encountered: {sic_code}. "
            "Add to SIC_Industry table for data remediation."
        )
        return (
            "unmapped_sic_code",
            f"No sector/industry mapping found for SIC code {sic_code}. Please select manually."
        )
    
    return ("success", None)


def check_company_exists(conn, company_house_number, website):
    """
    Check if a company already exists in the Company table.
    
    Args:
        conn: psycopg2 connection
        company_house_number (str): Companies House number
        website (str): Company website URL
    
    Returns:
        dict | None: Company data if exists, None if not found
    """
    # Normalize website for comparison (strip protocol and trailing slash)
    normalized_website = website.lower().strip()
    for prefix in ['https://', 'http://', 'www.']:
        if normalized_website.startswith(prefix):
            normalized_website = normalized_website[len(prefix):]
    normalized_website = normalized_website.rstrip('/')
    
    with conn.cursor() as cursor:
        query = """
            SELECT 
                c."CompanyID",
                c."CompanyName",
                c."CompanyWebSite",
                c."CompanyHouseNumber",
                c."SectorID",
                s."SectorDescription" as sector,
                c."IndustryID",
                i."IndustryDescription" as industry,
                c."NumberOfEmployeeID" as "number_of_employee_id",
                COALESCE(noe."NumberOfEmployeeDescription", '') as "employee_range",
                c."MetricRevenueID" as "metric_revenue_id",
                COALESCE(mr."MetricRevenueDescription", '') as "revenue_range"
            FROM "Company" c
            LEFT JOIN "Sector" s ON c."SectorID" = s."SectorID"
            LEFT JOIN "Industry" i ON c."IndustryID" = i."IndustryID"
            LEFT JOIN "NumberOfEmployee" noe ON c."NumberOfEmployeeID" = noe."NumberOfEmployeeID"
            LEFT JOIN "MetricRevenue" mr ON c."MetricRevenueID" = mr."MetricRevenueID"
            WHERE c."CompanyHouseNumber" = %s
              AND (
                  LOWER(REGEXP_REPLACE(REGEXP_REPLACE(c."CompanyWebSite", '^https?://', ''), '^www\\.', '')) = %s
                  OR LOWER(REGEXP_REPLACE(REGEXP_REPLACE(c."CompanyWebSite", '^https?://', ''), '^www\\.', '')) = %s
              )
            LIMIT 1
        """
        cursor.execute(query, (company_house_number, normalized_website, normalized_website.rstrip('/')))
        
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        result = dict(zip(columns, row))
        
        logging.info(
            f"Company already exists: CompanyID={result['CompanyID']}, "
            f"Name='{result['CompanyName']}', Website='{result['CompanyWebSite']}'"
        )
        
        return result
