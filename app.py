# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Title of the web app
# # st.title("Simple Streamlit Web App")

# # # Input text box for the user's name
# # name = st.text_input("Enter your name:")

# # # Button to trigger the greeting message
# # if st.button("Greet"):
# #     st.write(f"Hello, {name}!")


# # # Title of the web app
# # st.title("Simple Data Visualization with Streamlit")

# # # Generate a random dataset
# # np.random.seed(42)
# # data = pd.DataFrame({
# #     'x': np.random.randn(100),
# #     'y': np.random.randn(100)
# # })

# # # Display the dataset as a table
# # st.write("Here is the dataset:")
# # st.dataframe(data)

# # # Plotting the dataset
# # st.write("Here is a scatter plot of the dataset:")
# # fig, ax = plt.subplots()
# # ax.scatter(data['x'], data['y'])
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_title('Scatter Plot of Random Data')

# # # Display the plot
# # st.pyplot(fig)



# import streamlit as st
# import pandas as pd
# import gdown

# # URL of the Google Drive file
# file_id = '1INqhOyGt4LKM8Fiooj8aOvIqJDK-pCoK'
# gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# # Function to fetch and read the CSV data
# @st.cache
# def load_data(url):
#     output = 'purchase_m1.csv'
#     gdown.download(url, output, quiet=False)
#     return pd.read_csv(output)

# # Load the data
# df = load_data(gdrive_url)

# # Display the data
# st.title("CSV Data from Google Drive")
# st.write("Here is the dataset:")
# st.dataframe(df)

# st.write("--------shape-----------")
# st.write(df.shape)
# st.write("First 5 rows of the dataset:")
# st.dataframe(df.head(5))

import streamlit as st
import pandas as pd
import gdown

# URL of the Google Drive file
# file_id = '1INqhOyGt4LKM8Fiooj8aOvIqJDK-pCoK'
file_id = '1dHebTpV8IlzXdmGebg53-K93RlhP-KWd'
gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# Function to fetch and read the CSV data
def load_data(url):
    # output = 'purchase_m1.csv'
    output = '15_count.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Load the data
df = load_data(gdrive_url)

# Display the DataFrame shape and head(5) in the console
print(df.shape)
print(df.head(5))

# Display the head(5) of the DataFrame in Streamlit
st.write("### DataFrame Head (first 5 rows)")
st.write(df.head(5))
