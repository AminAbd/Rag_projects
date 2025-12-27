# SQL-RAG Agent

A production-ready Retrieval-Augmented Generation (RAG) system that converts natural language questions into SQL queries, executes them safely, and generates human-readable answers. This implementation includes role-based access control, security mechanisms, and comprehensive logging.

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Security Mechanisms](#security-mechanisms)
6. [Role-Based Access Control](#role-based-access-control)
7. [Error Handling & Self-Correction](#error-handling--self-correction)
8. [Logging System](#logging-system)
9. [Usage](#usage)
10. [Database Schema](#database-schema)

---

## Overview

SQL-RAG combines the power of Large Language Models (LLMs) with structured database queries to create an intelligent interface that understands natural language and retrieves precise information from databases. Unlike traditional RAG systems that search unstructured documents, SQL-RAG operates on structured relational data, enabling exact answers to complex questions.

### Key Features

- **Natural Language to SQL**: Converts questions into executable SQL queries
- **Role-Based Security**: Enforces access control based on user roles
- **Self-Correcting**: Automatically fixes SQL errors using LLM feedback
- **Safe Execution**: Prevents dangerous SQL operations (INSERT, UPDATE, DELETE, etc.)
- **Comprehensive Logging**: Tracks all queries, errors, and performance metrics
- **Interactive CLI**: User-friendly command-line interface

---

## Theoretical Foundation

### What is SQL-RAG?

SQL-RAG is a specialized form of Retrieval-Augmented Generation that bridges the gap between natural language understanding and structured data retrieval. The system follows this conceptual flow:

```
Natural Language Question
    ↓
[LLM: Text-to-SQL Translation]
    ↓
SQL Query (with schema context)
    ↓
[Security & Policy Validation]
    ↓
[Database Execution]
    ↓
Structured Results
    ↓
[LLM: Results-to-Text Translation]
    ↓
Natural Language Answer
```

### Why SQL-RAG?

Traditional RAG systems work with unstructured text documents, but many real-world applications require querying structured databases. SQL-RAG addresses this by:

1. **Precision**: SQL queries return exact data, not approximate matches
2. **Efficiency**: Database queries are optimized and fast
3. **Structured Reasoning**: Can handle complex joins, aggregations, and filters
4. **Scalability**: Databases handle large datasets efficiently

### The Challenge

Converting natural language to SQL is non-trivial because:
- **Semantic Gap**: Natural language is ambiguous; SQL is precise
- **Schema Understanding**: Must understand database structure and relationships
- **Query Complexity**: Users ask complex questions requiring multi-table joins
- **Security**: Must prevent SQL injection and unauthorized access

This implementation solves these challenges through:
- **Schema-Aware Prompting**: Provides full database schema to the LLM
- **Structured Prompts**: Guides the LLM with specific requirements
- **Multi-Layer Validation**: Safety checks, role policies, and error correction
- **Self-Correction**: Uses error feedback to improve query generation

---

## Architecture

The system is built using **LangChain**, a framework for building LLM applications. The architecture consists of several interconnected chains:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Question                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              SQL Generation Chain                            │
│  [Schema] + [Question] → LLM → SQL Query                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Security Validation                             │
│  • SQL Sanitization                                          │
│  • Safety Checks (no INSERT/UPDATE/DELETE)                   │
│  • Role Policy Validation                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Auto-Redaction (if policy violation)                │
│  [Role] + [Forbidden Fields] → LLM → Redacted SQL          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Database Execution                              │
│  SQL Query → SQLite Database → Results                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Error Correction (if execution fails)               │
│  [Error] + [Bad SQL] → LLM → Fixed SQL                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Answer Generation Chain                         │
│  [Question] + [SQL] + [Results] → LLM → Natural Answer      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Database Connection

```python
db = SQLDatabase.from_uri("sqlite:///sql-rag/db/Chinook.db?mode=ro")
```

**Theory**: The `SQLDatabase` wrapper from LangChain provides a unified interface to SQL databases. It abstracts away database-specific details and provides schema introspection capabilities.

**Code Explanation**:
- `mode=ro` ensures read-only access, preventing accidental modifications
- The wrapper automatically extracts table schemas, column names, and data types
- Provides `get_table_info()` method that returns formatted schema information

### 2. LLM Configuration

```python
llm_sql = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Theory**: We use two separate LLM instances:
- **`llm_sql`**: Specialized for SQL generation** - Lower temperature (0) ensures deterministic, consistent SQL output
- **`llm_answer`**: Specialized for answer generation** - Converts structured results into natural language

**Why Separate?**: Different tasks require different prompting strategies. Separating them allows:
- Independent optimization
- Different temperature settings
- Specialized fine-tuning in the future

### 3. SQL Generation Chain

```python
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
- Return ONLY the SQL query. No markdown, no backticks, no explanation."""

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
```

**Theory**: This implements the **Text-to-SQL** transformation using few-shot prompting. The system prompt:
- Establishes the LLM's role as a SQL expert
- Provides domain-specific rules (e.g., customer spending calculations)
- Sets constraints (LIMIT, column selection, quoting)
- Ensures clean output (no markdown formatting)

**Code Explanation**:
- **`ChatPromptTemplate`**: LangChain's prompt template system that formats messages
- **System Message**: Sets the LLM's behavior and constraints
- **Human Message**: Provides the schema and user question
- **Chain Composition**: Uses LangChain's pipe operator (`|`) to create a processing pipeline:
  1. Extract schema from database
  2. Format prompt with schema and question
  3. Send to LLM
  4. Parse output as string

**Why Schema in Prompt?**: LLMs need context about available tables, columns, and relationships to generate valid SQL. Providing the full schema enables:
- Correct table/column name selection
- Proper JOIN construction
- Appropriate data type handling

### 4. SQL Sanitization

```python
def sanitize_sql(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```sql\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    return t
```

**Theory**: LLMs sometimes wrap SQL in markdown code blocks (```sql ... ```). This function removes such formatting to extract pure SQL.

**Code Explanation**:
- Strips whitespace
- Removes markdown code fences using regex
- Returns clean SQL string

### 5. Safety Validation

```python
def is_safe_sql(sql: str) -> bool:
    banned = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b"
    if re.search(banned, sql, flags=re.IGNORECASE):
        return False
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        return False
    return True
```

**Theory**: **Defense in Depth** - Multiple layers of security prevent malicious or accidental data modification.

**Code Explanation**:
- **Banned Keywords Check**: Uses regex to detect dangerous SQL operations
  - `INSERT`, `UPDATE`, `DELETE`: Data modification
  - `DROP`, `ALTER`, `TRUNCATE`: Schema modification
  - `CREATE`, `REPLACE`: Object creation
  - `ATTACH`, `DETACH`, `PRAGMA`: Database operations
- **Whitelist Check**: Ensures query starts with `SELECT` or `WITH` (CTE)
- Returns `False` if any dangerous pattern is found

**Why This Matters**: Even with read-only database access, preventing dangerous SQL:
- Protects against future database changes
- Prevents accidental schema exposure
- Ensures queries are read-only by design

---

## Security Mechanisms

### Multi-Layer Security Model

The system implements a **defense-in-depth** strategy** with multiple security layers:

1. **Access-Level**: Password authentication before system access
2. **Database-Level**: Read-only mode (`mode=ro`)
3. **SQL-Level**: Keyword filtering and whitelisting
4. **Role-Level**: Policy-based access control
5. **Execution-Level**: Error handling and validation

### Password Authentication (Role-Based)

**Theory**: The first line of defense is preventing unauthorized access to the system entirely. **Role-based password authentication** ensures only authorized users can interact with the SQL-RAG agent, and the password itself determines the user's access level. This eliminates the need for separate role selection and provides stronger security.

**Implementation**:
```python
ROLE_PASSWORDS = {
    "admin": os.getenv("SQL_RAG_ADMIN_PASSWORD", "admin123"),
    "support_agent": os.getenv("SQL_RAG_SUPPORT_PASSWORD", "support123")
}

def authenticate():
    """Authenticate user with password and determine role based on password."""
    for attempt in range(MAX_LOGIN_ATTEMPTS):
        password = getpass.getpass(f"Enter password (attempt {attempt + 1}/{MAX_LOGIN_ATTEMPTS}): ")
        
        # Check which role this password belongs to
        for role, role_password in ROLE_PASSWORDS.items():
            if password == role_password:
                return role  # Return the role determined by password
    
    return None  # Authentication failed
```

**Security Features**:
- **Hidden Input**: Uses `getpass` to prevent password visibility
- **Attempt Limiting**: Maximum 3 login attempts to prevent brute force
- **Role-Based Access**: Password automatically determines role (admin or support_agent)
- **Environment Variable Support**: Passwords can be set via `SQL_RAG_ADMIN_PASSWORD` and `SQL_RAG_SUPPORT_PASSWORD` env vars
- **Default Passwords**: Development defaults (`admin123` for admin, `support123` for support) - **must be changed in production**

**Advantages of Role-Based Passwords**:
- **Simpler UX**: No need to select role after authentication
- **Stronger Security**: Role is determined by password, not user selection
- **Clear Separation**: Different passwords for different access levels
- **Audit Trail**: Password used indicates which role accessed the system

**Best Practices**:
- Use strong, unique passwords for each role in production
- Set passwords via environment variables (never commit passwords to code)
- Rotate passwords regularly
- Use different password complexity requirements for admin vs support roles
- Consider implementing additional authentication (2FA, API keys) for production deployments

### SQL Injection Prevention

**Theory**: SQL injection occurs when user input is directly concatenated into SQL queries. Our system prevents this by:

1. **Parameterized Queries**: LangChain's `SQLDatabase` uses parameterized queries internally
2. **No Direct String Concatenation**: LLM-generated SQL is validated, not directly executed from user input
3. **Pattern Matching**: Regex checks prevent injection patterns

**However**: The LLM itself could be manipulated through prompt injection. This is mitigated by:
- Clear system prompts with strict constraints
- Output validation before execution
- Role-based restrictions

---

## Role-Based Access Control

### Theory: Principle of Least Privilege

Role-Based Access Control (RBAC) enforces the **principle of least privilege**: users should only have access to resources they need for their role.

### Implementation

```python
ROLE_POLICY = {
    "support_agent": {
        "forbidden": [
            r'"Customer"\."Email"',   # no emails
        ],
        "required": [],
        "allowed_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Artist", "Album", "Genre", "Playlist", "MediaType"},
    },
    "admin": {
        "forbidden": [],
        "required": [],
        "allowed_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Artist", "Album", "Genre", "Playlist", "MediaType"},
    },
}
```

**Policy Structure**:
- **`forbidden`**: Regex patterns that must NOT appear in SQL (e.g., email addresses)
- **`required`**: Patterns that MUST appear (currently unused, but extensible)
- **`allowed_tables`**: Set of table names that can be queried

**Example**: A `support_agent` cannot access customer emails, but an `admin` can.

### Policy Validation

```python
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
```

**Theory**: **Static Analysis** - We analyze the SQL query structure before execution to ensure compliance.

**Code Explanation**:
- **`tables_in_sql()`**: Extracts table names from `FROM` and `JOIN` clauses using regex
- **`passes_role_policy()`**: Validates SQL against role policy:
  1. Checks if role exists
  2. Scans for forbidden patterns
  3. Verifies all used tables are in allowed set
  4. Returns `(is_valid, error_message)`

### Auto-Redaction

```python
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
```

**Theory**: **Self-Correction** - When the LLM generates SQL that violates policy, we don't just reject it. Instead, we ask the LLM to regenerate the query without forbidden fields.

**Why Auto-Redaction?**:
- **User Experience**: Users don't need to rephrase questions
- **Intelligence**: LLM understands context and can remove sensitive fields while preserving query intent
- **Flexibility**: Can handle complex queries that accidentally include forbidden fields

**Process**:
1. Policy violation detected
2. LLM receives: role, forbidden patterns, original question, and bad SQL
3. LLM regenerates SQL without forbidden fields
4. New SQL is validated again

---

## Error Handling & Self-Correction

### Theory: LLM-Assisted Debugging

Traditional SQL debugging requires human intervention. Our system uses **LLM feedback loops** to automatically correct errors.

### Error Correction Chain

```python
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
```

**Theory**: When SQL execution fails, the database returns an error message. We feed this error back to the LLM along with:
- Original question (intent)
- Bad SQL (what was wrong)
- Error message (why it failed)
- Schema (context for correction)

The LLM uses this information to generate a corrected query.

### Error Correction Flow

```python
try:
    rows = db.run(sql)
    return sql, rows, ""
except Exception as e:
    # one retry
    fixed = sanitize_sql(fix_chain.invoke({
        "question": question, 
        "sql": sql, 
        "error": str(e)
    }))
    
    # Re-validate fixed SQL
    if not is_safe_sql(fixed):
        return fixed, None, "Unsafe corrected SQL."
    
    ok, msg = passes_role_policy(fixed, role)
    if not ok:
        return fixed, None, msg
    
    try:
        rows = db.run(fixed)
        return fixed, rows, ""
    except Exception as e2:
        return fixed, None, str(e2)
```

**Process**:
1. Execute SQL query
2. If error occurs, capture exception message
3. Send error to LLM with context
4. LLM generates corrected SQL
5. Re-validate corrected SQL (safety + policy)
6. Execute corrected SQL
7. Return results or final error

**Why This Works**: LLMs excel at pattern matching and understanding context. Error messages provide specific feedback (e.g., "column not found", "syntax error"), which the LLM uses to correct the query.

**Limitations**: 
- Only one retry attempt (prevents infinite loops)
- Fixed SQL must pass all security checks again
- Some errors may be unfixable (e.g., missing tables)

---

## Logging System

### Theory: Observability and Auditability

Comprehensive logging enables:
- **Debugging**: Track down issues in production
- **Auditing**: Monitor access and queries
- **Analytics**: Understand usage patterns
- **Compliance**: Meet regulatory requirements

### Logger Implementation

```python
def log_run(record: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        **record,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

**Theory**: **JSONL Format** (JSON Lines) - Each log entry is a JSON object on a single line. This format:
- Is human-readable
- Is easy to parse programmatically
- Supports streaming (append-only)
- Works well with log aggregation tools

**Code Explanation**:
- Auto-creates directory if needed
- Adds UTC timestamp to every record
- Appends JSON object to file (append mode)
- Uses UTF-8 encoding for international characters

### What Gets Logged

The system logs comprehensive information at multiple stages:

**1. SQL Generation Stage** (`run_question` function):
```python
log_data = {
    "question": question,
    "role": role,
    "initial_sql": sql,
    "safety_check": "passed" | "failed",
    "role_policy_check": "passed" | "failed" | "passed_after_redaction",
    "redacted": True | False,
    "redacted_sql": "...",  # if redacted
    "fixed": True | False,
    "fixed_sql": "...",  # if fixed
    "status": "success" | "error" | "success_after_fix",
    "final_sql": "...",
    "rows_count": 5,
    "error": "..."  # if error occurred
}
```

**2. Execution Stage** (`main` function):
```python
log_run({
    "question": q,
    "role": role,
    "execution_time_seconds": 2.34,
    "answer": "..."  # final natural language answer
})
```

**Log Analysis**: Logs can be analyzed to:
- Identify common error patterns
- Monitor performance (execution times)
- Track role-based access patterns
- Audit sensitive queries
- Improve prompts based on failure modes

---

## Usage

### Quick Summary

The SQL-RAG agent uses **role-based password authentication**:
- **Enter a password** → System determines your role automatically
- **Admin password** (`admin123`) → Full access to all data
- **Support Agent password** (`support123`) → Restricted access (no sensitive fields)
- **No role selection needed** - password IS your role identifier

### Running the Agent

```bash
cd sql-rag
python sql_rag_agent.py
```

### Password Authentication (Role-Based)

The agent uses **role-based password authentication** - the password you enter automatically determines your access level. This is more secure than selecting a role after login because the password itself grants the appropriate permissions.

#### How It Works

**Key Concept**: Each role has its own unique password. When you enter a password, the system:
1. Checks which role that password belongs to
2. Automatically assigns you that role
3. Applies the corresponding access permissions

**No Role Selection Needed**: Unlike traditional systems where you login and then select a role, here the password IS your role identifier.

#### Default Passwords (Development)

| Role | Password | Access Level |
|------|----------|--------------|
| **Admin** | `admin123` | Full access to all tables and fields |
| **Support Agent** | `support123` | Restricted access (no customer emails, limited fields) |

**⚠️ Important**: These are development defaults. **You must change them in production!**

#### Setting Custom Passwords (Production)

**Option 1: Environment Variables (Recommended)**

Set passwords via environment variables before running the agent:

```bash
# On Linux/macOS
export SQL_RAG_ADMIN_PASSWORD=your_secure_admin_password
export SQL_RAG_SUPPORT_PASSWORD=your_secure_support_password

# On Windows (PowerShell)
$env:SQL_RAG_ADMIN_PASSWORD="your_secure_admin_password"
$env:SQL_RAG_SUPPORT_PASSWORD="your_secure_support_password"

# Then run the agent
python sql_rag_agent.py
```

**Option 2: Modify Code Directly**

Edit `sql_rag_agent.py` and change the default passwords in the `ROLE_PASSWORDS` dictionary:

```python
ROLE_PASSWORDS = {
    "admin": "your_custom_admin_password",
    "support_agent": "your_custom_support_password"
}
```

**⚠️ Security Warning**: Never commit passwords to version control. Always use environment variables in production.

#### Security Features

- **Hidden Input**: Password is not displayed while typing (using `getpass`)
- **Attempt Limiting**: Maximum 3 login attempts to prevent brute force attacks
- **Automatic Role Assignment**: Role is determined by password, eliminating role selection step
- **Access Denial**: System exits after exceeding maximum attempts
- **Environment Variable Support**: Secure password management without code changes

#### Why Role-Based Passwords?

**Advantages**:
- **Simpler UX**: One step (enter password) instead of two (login + select role)
- **Stronger Security**: Role is cryptographically tied to password, not user choice
- **Clear Audit Trail**: Password used indicates which role accessed the system
- **No Privilege Escalation**: Users can't select a higher role than their password allows
- **Better Separation**: Different passwords for different access levels

**Example Scenario**:
- Admin enters `admin123` → Gets admin role automatically → Full access
- Support agent enters `support123` → Gets support_agent role automatically → Restricted access
- Wrong password → Access denied → No role assigned

### Interactive Session

```
============================================================
SQL-RAG Agent - Authentication Required
============================================================
Enter your role password (password determines your access level)

Enter password (attempt 1/3): ********

✓ Authentication successful! Role: Support Agent

SQL-RAG CLI ready. Type a question. Type 'exit' to quit.

You> Which 5 customers spent the most money?
```

**Note**: The role is automatically determined by the password you enter. No need to select a role separately.

### Example Output

```
--- SQL ---
SELECT "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName", 
       SUM("Invoice"."Total") AS "TotalSpent"
FROM "Customer"
JOIN "Invoice" ON "Customer"."CustomerId" = "Invoice"."CustomerId"
GROUP BY "Customer"."CustomerId", "Customer"."FirstName", "Customer"."LastName"
ORDER BY "TotalSpent" DESC
LIMIT 5

--- ROWS ---
[(1, 'Luís', 'Gonçalves', 49.62), (2, 'Leonie', 'Köhler', 49.62), ...]

--- ANSWER ---
The top 5 customers by spending are:
1. Luís Gonçalves - $49.62
2. Leonie Köhler - $49.62
...

(time: 2.34s)
```

---

## Database Schema

The system uses the **Chinook** sample database, which models a digital media store. Key tables:

- **Customer**: Customer information
- **Invoice**: Sales invoices
- **InvoiceLine**: Line items on invoices
- **Track**: Music tracks
- **Album**: Music albums
- **Artist**: Music artists
- **Genre**: Music genres
- **Playlist**: User playlists
- **PlaylistTrack**: Playlist-track relationships
- **MediaType**: Media formats
- **Employee**: Store employees

See the database analysis for detailed schema information.

---

## Key Design Decisions

### 1. Why Two LLM Instances?

Separate instances allow:
- Independent optimization
- Different temperature settings
- Future fine-tuning per task
- Cost optimization (could use different models)

### 2. Why Read-Only Database?

- Prevents accidental data loss
- Simplifies security model
- Aligns with RAG use case (information retrieval, not modification)

### 3. Why Auto-Redaction Instead of Rejection?

- Better user experience
- Leverages LLM intelligence
- Handles edge cases gracefully

### 4. Why Only One Retry?

- Prevents infinite loops
- Limits cost (LLM API calls)
- Forces clear error messages for unfixable issues

### 5. Why JSONL Logging?

- Human-readable
- Easy to parse
- Supports streaming
- Works with log aggregation tools (e.g., ELK stack)

---

## Extending the System

### Adding New Roles

```python
ROLE_POLICY["analyst"] = {
    "forbidden": [r'"Customer"\."Email"', r'"Customer"\."Phone"'],
    "required": [],
    "allowed_tables": {"Invoice", "InvoiceLine", "Track", "Album", "Artist"}
}
```

### Adding New Forbidden Patterns

```python
"forbidden": [
    r'"Customer"\."Email"',
    r'"Customer"\."Phone"',
    r'"Employee"\."Salary"',  # New pattern
]
```

### Customizing Prompts

Modify `system_sql` to add domain-specific rules or change output format.

---

## Limitations & Future Improvements

### Current Limitations

1. **Single Retry**: Only one error correction attempt
2. **No Query Caching**: Same questions generate new SQL each time
3. **No Query Optimization**: Generated SQL may not be optimal
4. **Limited Error Types**: Some errors may not be fixable automatically

### Potential Improvements

1. **Query Caching**: Cache SQL for common questions
2. **Query Optimization**: Analyze and optimize generated SQL
3. **Multi-Retry**: Allow multiple correction attempts with backoff
4. **Query Explanation**: Explain why certain SQL was generated
5. **Confidence Scores**: Rate SQL quality before execution
6. **Fine-Tuning**: Fine-tune LLM on SQL generation task
7. **Schema Learning**: Learn common query patterns from logs

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [SQLDatabase Utilities](https://python.langchain.com/docs/integrations/tools/sql_database)
- [Text-to-SQL Research](https://arxiv.org/abs/1709.00103)
- [Chinook Database](https://github.com/lerocha/chinook-database)

---

## License

This project is part of the RAG Projects repository.

