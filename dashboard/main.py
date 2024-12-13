"""Main Page."""

import streamlit as st

st.set_page_config("Dashboard")
st.title("Dashboard")
st.page_link(st.Page("pages/profiles.py"), label="## Profiles")
st.page_link(st.Page("pages/energetics.py"), label="## Energetics")
