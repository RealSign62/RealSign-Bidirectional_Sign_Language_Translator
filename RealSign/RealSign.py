import streamlit as st
import pandas as pd
import numpy as np


# ------ Code snippet to switch between pages ---------
def switch_page(page_name: str):
    from streamlit import _RerunData, _RerunException
    from streamlit.source_util import get_pages

    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages("RealSign.py")
    # pages = get_pages("streamlit_app.py")  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise _RerunException(
                _RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [standardize_name(config["page_name"]) for config in pages.values()]

    raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")

# ------- Title of the app -------
st.title('RealSign: Real-Time Instantaneous Bidirectional Indian Sign Language Translation')


# ------- Switching page functionality -------
ssl = st.button('Speech to Sign Language')
if ssl:
    switch_page("ssl")

slr = st.button('Sign Language to Speech/Text')
if slr:
    switch_page("slr")




