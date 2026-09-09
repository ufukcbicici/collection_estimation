"""Paths and corpus-level constants.

Everything that knows where data lives is here, so no other module hardcodes a path.
"""
from __future__ import annotations

import os

# Repository root: this file is at <root>/src/collection_estimation/config.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
INTERIM = os.path.join(DATA, "interim")

#: The corpus: one Excel workbook per business day. A COPY of the output of
#: `gen_collections_corpus.py` in the cockpit repo, gitignored here because it is 5.6 MB of
#: generated data reproducible from a seed.
CORPUS_DIR = os.path.join(RAW, "SYNTHETIC_ey_tahsilat_20250602_20260828")

#: Ground truth describing the corpus: planted patterns, customer archetypes and
#: lifecycles, the hidden weekly billing series, benchmark results.
#: NEVER use its fields as model features — they exist only because the data is synthetic.
ORACLE = os.path.join(RAW, "collections_corpus.json")

#: Tidy collections table written by `ingest.py`, read by everything downstream.
COLLECTIONS = os.path.join(INTERIM, "collections.csv")

# --------------------------------------------------------------------------------------
# Corpus layout. See `daily_collections_corpus.md` in the cockpit repo.
# --------------------------------------------------------------------------------------

#: Header is on row 2; row 1 is a merged "MERCURY COLLECTIONS" title.
HEADER_ROW = 2
FIRST_DATA_ROW = 3

#: EY legal entities. A row belongs to one; subtotal rows carry the label but no code.
ENTITY_LABEL = {"TR02": "Güney", "TR04": "Eykf", "TR05": "Av.Ort",
                "TR06": "Bey", "TR91": "Kuzey"}

#: `collection_group` as it appears in EY's imported schema.
ENTITY_GROUP = {"TR02": "G", "TR04": "E", "TR05": "Ao", "TR06": "B", "TR91": "K"}

#: Column G carries an invoice number, or one of these Turkish status notes instead. Such
#: rows are UNAPPLIED CASH: money received but not yet matched to a receivable. They are
#: part of the day's total and must not be dropped.
STATUS_NOTES = ("Ödeme Detayı Araştırılıyor", "Eğitim Ödemesi")

CURRENCIES = ("TRY", "EUR", "USD", "GBP")
