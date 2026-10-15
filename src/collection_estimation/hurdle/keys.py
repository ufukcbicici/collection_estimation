"""The key every later step groups and joins on: company code + customer.

A customer who pays two companies gets two keys, because the forecast is per company.
"""
import pandas as pd

# Characters that are invisible but not removed by str.strip(): the zero-width space and the
# byte-order mark (which some CSV exports put at the very start of a file).
INVISIBLE_CHARACTERS = ["​", "﻿"]


def normalise_customer_id(raw: str) -> str:
    """One spelling per customer: '  0011226452 ' -> '11226452'; ' ab12 ' -> 'AB12'.

    Used for every source (daily files, AR ledger), so the same customer gets the same ID.
    Raises ValueError for a value that is not text or is empty.
    """
    if not isinstance(raw, str):
        raise ValueError(f"customer ID {raw!r} is not text")
    text = raw
    for character in INVISIBLE_CHARACTERS:
        text = text.replace(character, "")
    text = text.strip().upper()
    if text == "":
        raise ValueError("customer ID is empty")
    # isascii() as well, because isdigit() also accepts characters such as '²'.
    if text.isdigit() and text.isascii():
        # SAP stores numeric customer numbers zero-padded to 10 digits, but reports often
        # show them without the zeros; drop them so both spellings give the same ID.
        text = text.lstrip("0") or "0"
    return text


def add_keys(collections: pd.DataFrame, known_companies: list[str],
             customer_map: dict[str, str] | None = None) -> pd.DataFrame:
    """The payment lines with two new columns: `customer_id` and `key` (e.g. 'TR02-11226452').

    `known_companies`: the company codes we expect; any other code raises ValueError.
    `customer_map`: optional {customer ID: master ID}, e.g. to merge two accounts of one
    customer; applied after normalising. The original `customer` column is kept.
    """
    check_company_codes(collections["company_code"], known_companies)

    customer_ids = collections["customer"].map(normalise_customer_id)
    if customer_map:
        # Normalise the map too, so '0011226452' in the map matches '11226452' in the data.
        clean_map = {normalise_customer_id(raw): normalise_customer_id(master)
                     for raw, master in customer_map.items()}
        customer_ids = customer_ids.map(lambda customer_id: clean_map.get(customer_id, customer_id))

    with_keys = collections.copy()
    # Text columns use pandas' generic object type throughout the package (as the readers'
    # do); left alone, pandas would give these two its own text type.
    with_keys["customer_id"] = customer_ids.astype(object)
    with_keys["key"] = (with_keys["company_code"] + "-" + with_keys["customer_id"]).astype(object)
    return with_keys


def check_company_codes(company_codes: pd.Series, known_companies: list[str]) -> None:
    """Raise ValueError if any line carries a company code that is not in `known_companies`."""
    is_unknown = ~company_codes.isin(known_companies)
    if is_unknown.any():
        # dropna=False, so a missing code is named in the message too.
        lines_per_code = company_codes[is_unknown].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"unknown company codes (code: number of lines): {lines_per_code}; "
            f"known codes are {sorted(known_companies)}")
