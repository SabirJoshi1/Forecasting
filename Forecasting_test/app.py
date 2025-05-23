import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple Matplotlib Test App")

# Sample plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test Plot")

st.pyplot(fig)