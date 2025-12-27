import re
import time
import os
import getpass
from typing import Any, List, Tuple

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from logger import log_run

ROLE_POLICY = {
    "support_agent": {
        "forbidden": [
            r'"Customer"\."Email"',   # no emails
        ],
        "required": [
            # for sensitive queries, you can require aggregation only; keep minimal for now
        ],
        "allowed_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Artist", "Album", "Genre", "Playlist", "MediaType"},
    },
    "admin": {
        "forbidden": [],
        "required": [],
        "allowed_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Artist", "Album", "Genre", "Playlist", "MediaType"},
    },
}
########################################

load_dotenv()

# Role-based password configuration
# Passwords can be set via environment variables or use defaults below
# For production, use environment variables:
# export SQL_RAG_ADMIN_PASSWORD=your_admin_password
# export SQL_RAG_SUPPORT_PASSWORD=your_support_password
ROLE_PASSWORDS = {
    "admin": os.getenv("SQL_RAG_ADMIN_PASSWORD", "admin123"),
    "support_agent": os.getenv("SQL_RAG_SUPPORT_PASSWORD", "support123")
}
MAX_LOGIN_ATTEMPTS = 3

db = SQLDatabase.from_uri("sqlite:///sql-rag/db/Chinook.db?mode=ro")

llm_sql = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_sql = """You are a SQL expert. Given an input question, create a syntactically correct SQLite query to run.

Hard requirements:
- If the question is about customers and spending, return:
  "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName", "Customer"."Email"
  and the computed total spent as "TotalSpent".
- To compute spending, use "Invoice"."Total" and join "Customer" to "Invoice" on "CustomerId".

General requirements:
- Query for at most 5 results unless the user requests otherwise using LIMIT.
- Never query for all columns. Only select needed columns. Wrap identifiers in double quotes.
- Use only columns/tables from the provided schema.
- Return ONLY the SQL query. No markdown, no backticks, no explanation.
"""

prompt_sql = ChatPromptTemplate.from_messages([
    ("system", system_sql),
    ("human", "Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQLQuery:")
])

sql_chain = (
    {"schema": lambda _: db.get_table_info(), "question": lambda x: x["question"]}
    | prompt_sql
    | llm_sql
    | StrOutputParser()
)

def sanitize_sql(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```sql\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    return t

def is_safe_sql(sql: str) -> bool:
    banned = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b"
    if re.search(banned, sql, flags=re.IGNORECASE):
        return False
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        return False
    return True


def tables_in_sql(sql: str) -> set[str]:
    hits = re.findall(r'(?:FROM|JOIN)\s+"([^"]+)"', sql, flags=re.IGNORECASE)
    return set(hits)

def passes_role_policy(sql: str, role: str) -> tuple[bool, str]:
    policy = ROLE_POLICY.get(role)
    if not policy:
        return False, f"Unknown role: {role}"

    # forbidden patterns
    for pat in policy["forbidden"]:
        if re.search(pat, sql, flags=re.IGNORECASE):
            return False, f"Forbidden field for role '{role}'."

    # allowed tables
    used = tables_in_sql(sql)
    if not used.issubset(policy["allowed_tables"]):
        return False, f"SQL references tables not allowed for role '{role}': {used - policy['allowed_tables']}"

    return True, ""


fix_prompt = ChatPromptTemplate.from_messages([
    ("system", "Fix the SQLite query. Return ONLY the corrected SQL. No markdown."),
    ("human", "Schema:\n{schema}\n\nQuestion:\n{question}\n\nBad SQL:\n{sql}\n\nError:\n{error}\n\nCorrected SQL:")
])
fix_chain = (
    {"schema": lambda _: db.get_table_info(),
     "question": lambda x: x["question"],
     "sql": lambda x: x["sql"],
     "error": lambda x: x["error"]}
    | fix_prompt
    | llm_sql
    | StrOutputParser()
)

redact_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You must follow role-based access rules. "
     "Regenerate the SQL without any forbidden fields. Return ONLY SQL. No markdown."),
    ("human",
     "Role: {role}\nForbidden patterns: {forbidden}\n\nSchema:\n{schema}\n\nQuestion:\n{question}\n\nBad SQL:\n{sql}\n\nRegenerate SQL (no forbidden fields):")
])

redact_chain = (
    {"schema": lambda _: db.get_table_info(),
     "role": lambda x: x["role"],
     "forbidden": lambda x: x["forbidden"],
     "question": lambda x: x["question"],
     "sql": lambda x: x["sql"]}
    | redact_prompt
    | llm_sql
    | StrOutputParser()
)


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the rows. If rows are empty, say you don't know. Be concise."),
    ("human", "Question: {question}\nSQL: {sql}\nRows: {rows}\n\nAnswer:")
])
answer_chain = answer_prompt | llm_answer | StrOutputParser()

def run_question(question: str, role: str):
    raw_sql = sql_chain.invoke({"question": question})
    sql = sanitize_sql(raw_sql)
    
    log_data = {
        "question": question,
        "role": role,
        "initial_sql": sql,
        "safety_check": "pending",
        "role_policy_check": "pending",
        "redacted": False,
        "fixed": False,
        "status": "pending"
    }

    if not is_safe_sql(sql):
        log_data["safety_check"] = "failed"
        log_data["status"] = "error"
        log_data["error"] = "Unsafe SQL generated."
        log_run(log_data)
        return sql, None, "Unsafe SQL generated."
    
    log_data["safety_check"] = "passed"
    
    ok, msg = passes_role_policy(sql, role)
    if not ok:
        log_data["role_policy_check"] = "failed"
        log_data["role_policy_error"] = msg
     # Auto-redact and retry once (role-based)
        forbidden = ROLE_POLICY[role]["forbidden"]
        redacted_raw = redact_chain.invoke({
            "role": role,
            "forbidden": forbidden,
            "question": question,
            "sql": sql,
        })
        redacted_sql = sanitize_sql(redacted_raw)
        log_data["redacted"] = True
        log_data["redacted_sql"] = redacted_sql
        ok2, msg2 = passes_role_policy(redacted_sql, role)
        if not ok2:
            log_data["status"] = "error"
            log_data["error"] = msg2
            log_run(log_data)
            return redacted_sql, None, msg2

        sql = redacted_sql  # continue with safe SQL
        log_data["role_policy_check"] = "passed_after_redaction"

    else:
        log_data["role_policy_check"] = "passed"

    try:
        rows = db.run(sql)
        log_data["status"] = "success"
        log_data["final_sql"] = sql
        log_data["rows_count"] = len(str(rows).split('\n')) if rows else 0
        log_run(log_data)
        return sql, rows, ""
    except Exception as e:
        log_data["status"] = "error"
        log_data["error"] = str(e)
        # one retry
        fixed = sanitize_sql(fix_chain.invoke({"question": question, "sql": sql, "error": str(e)}))
        log_data["fixed"] = True
        log_data["fixed_sql"] = fixed
        if not is_safe_sql(fixed):
            log_data["safety_check"] = "failed_after_fix"
            log_data["error"] = "Unsafe corrected SQL."
            log_run(log_data)
            return fixed, None, "Unsafe corrected SQL."

        ok, msg = passes_role_policy(fixed, role)
        if not ok:
            log_data["role_policy_check"] = "failed_after_fix"
            log_data["error"] = msg
            log_run(log_data)
            return fixed, None, msg

        try:
            rows = db.run(fixed)
            log_data["status"] = "success_after_fix"
            log_data["final_sql"] = fixed
            log_data["rows_count"] = len(str(rows).split('\n')) if rows else 0
            log_run(log_data)
            return fixed, rows, ""
        except Exception as e2:
            log_data["status"] = "error_after_fix"
            log_data["error"] = str(e2)
            log_run(log_data)
            return fixed, None, str(e2)

def authenticate():
    """Authenticate user with password and determine role based on password."""
    print("=" * 60)
    print("SQL-RAG Agent - Authentication Required")
    print("=" * 60)
    print("Enter your role password (password determines your access level)\n")
    
    for attempt in range(MAX_LOGIN_ATTEMPTS):
        password = getpass.getpass(f"Enter password (attempt {attempt + 1}/{MAX_LOGIN_ATTEMPTS}): ")
        
        # Check which role this password belongs to
        for role, role_password in ROLE_PASSWORDS.items():
            if password == role_password:
                role_display = "Admin" if role == "admin" else "Support Agent"
                print(f"\n✓ Authentication successful! Role: {role_display}\n")
                return role
        
        # Password didn't match any role
        remaining = MAX_LOGIN_ATTEMPTS - (attempt + 1)
        if remaining > 0:
            print(f"✗ Incorrect password. {remaining} attempt(s) remaining.\n")
        else:
            print("✗ Maximum login attempts exceeded. Access denied.")
            return None
    
    return None

def main():
    # Require password authentication - password determines role
    role = authenticate()
    if not role:
        print("\nAccess denied. Exiting...")
        return
    
    print("SQL-RAG CLI ready. Type a question. Type 'exit' to quit.\n")

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        t0 = time.time()
        sql, rows, err = run_question(q, role)
        dt = time.time() - t0

        print("\n--- SQL ---")
        print(sql)

        if err:
            print("\n--- ERROR ---")
            print(err)
            print(f"\n(time: {dt:.2f}s)\n")
            # Error details already logged in run_question, just log execution time
            log_run({
                "question": q,
                "role": role,
                "execution_time_seconds": round(dt, 2)
            })
            continue

        print("\n--- ROWS ---")
        print(rows)

        final = answer_chain.invoke({"question": q, "sql": sql, "rows": rows})
        print("\n--- ANSWER ---")
        print(final)
        print(f"\n(time: {dt:.2f}s)\n")
        
        # Log final answer and execution time (SQL details already logged in run_question)
        log_run({
            "question": q,
            "role": role,
            "execution_time_seconds": round(dt, 2),
            "answer": final
        })

if __name__ == "__main__":
    main()
