"""
Database connection module for PostgreSQL RDS.
Supports multiple environments (dev, prod).
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text

# ====================== ENV HELPERS ======================
def _get_required_env(key, env_name=None):
    value = os.getenv(key)
    if value is None:
        env_context = f" for environment '{env_name}'" if env_name else ""
        raise ValueError(f"Required environment variable '{key}' is not set{env_context}")
    return value


# ====================== CONNECTION ======================
def rds_start_connection(env, port=None):
    try:
        # Load environment-specific configurations
        if env == "dev":
            host = _get_required_env("DB_DEV_HOST", env)
            dbname = _get_required_env("DB_DEV_NAME", env)
            user = _get_required_env("DB_DEV_USERNAME", env)
            password = _get_required_env("DB_DEV_PASSWORD", env)
            dbport = port if port else int(_get_required_env("DB_DEV_PORT", env))
            
        elif env == "prod":
            host = _get_required_env("DB_PROD_HOST", env)
            dbname = _get_required_env("DB_PROD_NAME", env)
            user = _get_required_env("DB_PROD_USERNAME", env)
            password = _get_required_env("DB_PROD_PASSWORD", env)
            dbport = port if port else int(_get_required_env("DB_PROD_PORT", env))
            
        else:
            raise ValueError("env must be 'dev' or 'prod'")

        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{dbport}/{dbname}"
        )

        # lightweight test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print("Connection successful.")
        return engine

    except Exception as e:
        logging.exception("Database connection failed")
        return None

# ====================== QUERIES ======================
def query_companies(query ,env="dev"):
    print(f"\nConnecting to {env.upper()} database...")
    
    try:
        engine = rds_start_connection(env)
        if engine is None:
            print("[FAIL] Connection failed")
            return None
        
        print("Executing query...")
        df = pd.read_sql_query(query, engine)
        
        print(f"Query executed successfully. Total: {len(df)} rows.")
        return df
        
    except Exception as e:
        print(f"[FAIL] Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None