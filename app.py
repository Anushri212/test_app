import streamlit as st

# Title of the web app
st.title("Simple Streamlit Web App")

# Input text box for the user's name
name = st.text_input("Enter your name:")

# Button to trigger the greeting message
if st.button("Greet"):
    st.write(f"Hello, {name}!")
