# .vulture_whitelist.py
"""Whitelist for vulture dead code detection."""

import streamlit as st
from utils.pdf_loader import load_pdf_from_upload, get_document_metadata

_ = st.session_state.pending_uploads
_ = load_pdf_from_upload
_ = get_document_metadata
