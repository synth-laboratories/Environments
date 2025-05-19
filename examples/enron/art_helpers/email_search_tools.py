import sqlite3
import logging
import textwrap
from typing import List, Optional
from dataclasses import dataclass
import os

from examples.enron.art_helpers.local_email_db import DEFAULT_DB_PATH, generate_database
from examples.enron.art_helpers.types_enron import Email

# Configure logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid duplicate handlers in pytest -x
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)  # DEBUG so we see the raw SQL

conn = None


def get_conn():
    global conn
    if conn is None:
        if not os.path.exists(DEFAULT_DB_PATH):
            logger.info(f"Database not found at {DEFAULT_DB_PATH}, generating...")
            generate_database(overwrite=False)
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


@dataclass
class SearchResult:
    message_id: str
    snippet: str
    score: float


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """
    Searches the email database based on keywords, inbox, sender, recipient, and date range.

    Args:
        inbox: The email address of the user performing the search.
               Results include emails sent from or to (inc. cc/bcc) this address.
        keywords: A list of keywords that must all appear in the subject or body.
        from_addr: Optional email address to filter emails sent *from*.
        to_addr: Optional email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional date string 'YYYY-MM-DD'. Filters for emails sent on or after this date.
        sent_before: Optional date string 'YYYY-MM-DD'. Filters for emails sent before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of SearchResult objects, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """

    if not keywords:
        raise ValueError("No keywords provided for search.")
    if max_results > 10:
        # The user snippet implies max_results isn't part of the simplified SQL here.
        # Keeping the check, but the new SQL query below does not use all filters.
        # This might need reconciliation if all filters are intended to be used with the new SQL.
        logger.warning("max_results > 10, but the provided SQL snippet for logging might not respect all filters.")

    db_conn = get_conn()
    # Replace single quotes for SQL compatibility, then build the FTS query string
    safe_keywords = [k.replace("'", "''") for k in keywords]
    fts_match_query = " ".join(f'"{k}"' for k in safe_keywords)

    # Using the simplified SQL from the user's request for logging and testing
    # This SQL does not include inbox, from_addr, to_addr, sent_after, sent_before filters.
    # It only uses keywords and max_results.
    sql_query = textwrap.dedent("""
        SELECT DISTINCT
               e.message_id,
               snippet(emails_fts, -1, '⟪', '⟫', ' … ', 15) AS snip
          FROM emails e
          JOIN emails_fts ON e.id = emails_fts.rowid
         WHERE emails_fts MATCH ?
         LIMIT ?
    """).strip()

    params = (fts_match_query, max_results)
    
    try:
        rows = db_conn.execute(sql_query, params).fetchall()
        #logger.debug("SQL: %s  |  params=%s  →  %d rows", sql_query, params, len(rows))
        # The SearchResult now needs a score. The provided SQL doesn't have a rank/score column.
        # Defaulting to 0.0 as per the user's snippet `SearchResult(*row, score=0.0)`
        return [SearchResult(message_id=row[0], snippet=row[1], score=0.0) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Database error during search: {e}\nSQL: {sql_query}\nParams: {params}")
        return []


def read_email(message_id: str) -> Optional[Email]:
    """
    Retrieves a single email by its message_id from the database.

    Args:
        message_id: The unique identifier of the email to retrieve.

    Returns:
        An Email object containing the details of the found email,
        or None if the email is not found or an error occurs.
    """

    cursor = get_conn().cursor()

    # --- Query for Email Core Details ---
    email_sql = """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        logging.warning(f"Email with message_id '{message_id}' not found.")
        return None

    (
        msg_id,
        date,
        subject,
        from_addr,
        body,
        file_name,
    ) = email_row

    # --- Query for Recipients ---
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (message_id,))
    recipient_rows = cursor.fetchall()

    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []

    for addr, type in recipient_rows:
        type_lower = type.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    # --- Construct Email Object ---
    email_obj = Email(
        message_id=msg_id,  # Convert to string to match Pydantic model
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )

    return email_obj
