# # # # # # import streamlit as st
# # # # # # import pandas as pd
# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt

# # # # # # # Title of the web app
# # # # # # st.title("Simple Streamlit Web App")

# # # # # # # Input text box for the user's name
# # # # # # name = st.text_input("Enter your name:")

# # # # # # # Button to trigger the greeting message
# # # # # # if st.button("Greet"):
# # # # # #     st.write(f"Hello, {name}!")


# # # # # # # Title of the web app
# # # # # # st.title("Simple Data Visualization with Streamlit")

# # # # # # # Generate a random dataset
# # # # # # np.random.seed(42)
# # # # # # data = pd.DataFrame({
# # # # # #     'x': np.random.randn(100),
# # # # # #     'y': np.random.randn(100)
# # # # # # })

# # # # # # # Display the dataset as a table
# # # # # # st.write("Here is the dataset:")
# # # # # # st.dataframe(data)

# # # # # # # Plotting the dataset
# # # # # # st.write("Here is a scatter plot of the dataset:")
# # # # # # fig, ax = plt.subplots()
# # # # # # ax.scatter(data['x'], data['y'])
# # # # # # ax.set_xlabel('X')
# # # # # # ax.set_ylabel('Y')
# # # # # # ax.set_title('Scatter Plot of Random Data')

# # # # # # # Display the plot
# # # # # # st.pyplot(fig)



# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import gdown

# # # # # # URL of the Google Drive file
# # # # # file_id = '1INqhOyGt4LKM8Fiooj8aOvIqJDK-pCoK'
# # # # # gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# # # # # # Function to fetch and read the CSV data
# # # # # @st.cache
# # # # # def load_data(url):
# # # # #     output = 'purchase_m1.csv'
# # # # #     gdown.download(url, output, quiet=False)
# # # # #     return pd.read_csv(output)

# # # # # # Load the data
# # # # # df = load_data(gdrive_url)

# # # # # # Display the data
# # # # # st.title("CSV Data from Google Drive")
# # # # # st.write("Here is the dataset:")
# # # # # st.dataframe(df)

# # # # # st.write("--------shape-----------")
# # # # # st.write(df.shape)
# # # # # st.write("First 5 rows of the dataset:")
# # # # # st.dataframe(df.head(5))

# # # # import streamlit as st
# # # # import pandas as pd
# # # # import gdown

# # # # # URL of the Google Drive file
# # # # # file_id = '1INqhOyGt4LKM8Fiooj8aOvIqJDK-pCoK'
# # # # file_id = '1dHebTpV8IlzXdmGebg53-K93RlhP-KWd'
# # # # gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# # # # # Function to fetch and read the CSV data
# # # # def load_data(url):
# # # #     # output = 'purchase_m1.csv'
# # # #     output = '15_count.csv'
# # # #     gdown.download(url, output, quiet=False)
# # # #     return pd.read_csv(output)

# # # # # Load the data
# # # # df = load_data(gdrive_url)

# # # # # Display the DataFrame shape and head(5) in the console
# # # # print(df.shape)
# # # # print(df.head(5))

# # # # # Display the head(5) of the DataFrame in Streamlit
# # # # st.write("### DataFrame Head (first 5 rows)")
# # # # st.write(df.head(5))

# # # ###################################

# # # import streamlit as st
# # # import pandas as pd
# # # import gdown
# # # import os

# # # # Function to fetch and read the CSV data from Google Drive
# # # def load_data_from_gdrive(file_id):
# # #     url = f'https://drive.google.com/uc?id={file_id}'
# # #     output = 'downloaded_data.csv'
# # #     if not os.path.exists(output):
# # #         gdown.download(url, output, quiet=False)
# # #     return pd.read_csv(output)

# # # # Page 1: Load dataset and preprocess data
# # # def page_load_and_preprocess():
# # #     st.title("LSTM Time Series Forecasting")
# # #     st.header("Step 1: Load Dataset")

# # #     # Input field to enter the Google Drive file ID
# # #     file_id = st.text_input("Enter the Google Drive file ID", placeholder="1dHebTpV8IlzXdmGebg53-K93RlhP-KWd")

# # #     if file_id:
# # #         try:
# # #             df = load_data_from_gdrive(file_id)
# # #             st.write("Data Preview:")
# # #             st.write(df.head())
# # #         except Exception as e:
# # #             st.write(f"Error loading data: {e}")
# # #     else:
# # #         st.write("Please enter a valid Google Drive file ID to load the dataset.")

# # # # Run the app
# # # if __name__ == "__main__":
# # #     page_load_and_preprocess()

# # ############################## main ########################################################################################

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import RobustScaler
# # from sklearn.model_selection import train_test_split
# # from keras.models import Sequential
# # from keras.layers import LSTM, Dropout, Dense, Bidirectional, Attention
# # from keras.callbacks import EarlyStopping, Callback
# # import matplotlib.pyplot as plt
# # import time
# # import os
# # import random
# # import gdown

# # # Custom callback to log epoch times
# # class EpochTimeCallback(Callback):
# #     def on_epoch_begin(self, epoch, logs=None):
# #         self.epoch_start_time = time.time()

# #     def on_epoch_end(self, epoch, logs=None):
# #         epoch_time = time.time() - self.epoch_start_time
# #         logs['epoch_time'] = epoch_time
# #         st.write(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")

# # # Function to create sequences
# # def create_sequences(data, sequence_length, output_sequence_length, target_column):
# #     X, y = [], []
# #     data_array = data.values
# #     total_length = sequence_length + output_sequence_length
# #     for i in range(len(data_array) - total_length + 1):
# #         seq_x = data_array[i:(i + sequence_length)]
# #         seq_y = data_array[i + sequence_length:(i + sequence_length + output_sequence_length), target_column]
# #         seq_y = seq_y.reshape(-1)
# #         X.append(seq_x)
# #         y.append(seq_y)
# #     return np.array(X), np.array(y)

# # # Function to preprocess data
# # def preprocess_data(df, misisdn_column, event_column_date, columns_to_scale, target_column, sequence_length, output_sequence_length):
# #     msisdns = df[misisdn_column].unique()
# #     train_msisdns, test_msisdns = train_test_split(msisdns, test_size=0.2, random_state=42)
# #     train_data = df[df[misisdn_column].isin(train_msisdns)]
# #     test_data = df[df[misisdn_column].isin(test_msisdns)]
# #     scaler = RobustScaler()
# #     train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
# #     test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])

# #     X_train_dict, y_train_dict = {}, {}
# #     for msisdn, group in train_data.groupby(misisdn_column):
# #         group = group.sort_values(event_column_date)
# #         lstm_data = group[columns_to_scale]
# #         if len(lstm_data) >= sequence_length + output_sequence_length:
# #             X_train, y_train = create_sequences(lstm_data, sequence_length, output_sequence_length, df.columns.get_loc(target_column))
# #             X_train_dict[msisdn] = X_train
# #             y_train_dict[msisdn] = y_train

# #     trainX = np.concatenate(list(X_train_dict.values()))
# #     trainY = np.concatenate(list(y_train_dict.values()))

# #     X_test_dict, y_test_dict = {}, {}
# #     for msisdn, group in test_data.groupby(misisdn_column):
# #         group = group.sort_values(event_column_date)
# #         lstm_data = group[columns_to_scale]
# #         if len(lstm_data) >= sequence_length + output_sequence_length:
# #             X_test, y_test = create_sequences(lstm_data, sequence_length, output_sequence_length, df.columns.get_loc(target_column))
# #             X_test_dict[msisdn] = X_test
# #             y_test_dict[msisdn] = y_test

# #     testX = np.concatenate(list(X_test_dict.values()))
# #     testY = np.concatenate(list(y_test_dict.values()))

# #     return trainX, trainY, testX, testY, scaler

# # # Function to fetch and read the CSV data from Google Drive
# # def load_data_from_gdrive(file_id):
# #     url = f'https://drive.google.com/uc?id={file_id}'
# #     output = 'downloaded_data.csv'
# #     if not os.path.exists(output):
# #         gdown.download(url, output, quiet=False)
# #     return pd.read_csv(output)

# # # Page 1: Load dataset and preprocess data
# # def page_load_and_preprocess():
# #     st.title("LSTM Time Series Forecasting")
# #     st.header("Step 1: Load Dataset")
    
# #     # Input field to enter the Google Drive file ID
# #     file_id = st.text_input("Enter the Google Drive file ID", placeholder="1dHebTpV8IlzXdmGebg53-K93RlhP-KWd")

# #     if file_id:
# #         try:
# #             df = load_data_from_gdrive(file_id)
# #             st.write("Data Preview:")
# #             st.write(df.head())

# #             # Proceed if data is loaded
# #             st.header("Step 2: Preprocess Data")
# #             misisdn_column = st.text_input("MSISDN Column", "Cust_Sub_Id")
# #             event_column_date = st.text_input("Date Column", "Event_Date")
# #             columns_to_scale = st.multiselect("Columns to Scale", df.columns.tolist(), default=['Total_Revenue', 'Data_Volume', 'OG_Call_Count'])
# #             df = df[[misisdn_column, event_column_date] + columns_to_scale]
# #             target_column = st.selectbox("Target Column", df.columns.tolist())
# #             sequence_length = st.number_input("Sequence Length", min_value=1, max_value=100, value=30)
# #             output_sequence_length = st.number_input("Output Sequence Length", min_value=1, max_value=10, value=1)

# #             if st.button("Preprocess Data"):
# #                 st.write("Starting data preprocessing...")  # Log message for preprocessing
# #                 trainX, trainY, testX, testY, scaler = preprocess_data(df, misisdn_column, event_column_date, columns_to_scale, target_column, sequence_length, output_sequence_length)
# #                 st.session_state['trainX'] = trainX
# #                 st.session_state['trainY'] = trainY
# #                 st.session_state['testX'] = testX
# #                 st.session_state['testY'] = testY
# #                 st.session_state['scaler'] = scaler
# #                 st.session_state['df'] = df
# #                 st.session_state['columns_to_scale'] = columns_to_scale
# #                 st.session_state['target_column'] = target_column
# #                 st.session_state['sequence_length'] = sequence_length
# #                 st.session_state['output_sequence_length'] = output_sequence_length
# #                 st.session_state['misisdn_column'] = misisdn_column
# #                 st.session_state['event_column_date'] = event_column_date

# #                 st.write("Data Preprocessing Completed")
# #                 st.write(f"Training Data Shape: {trainX.shape}")
# #                 st.write(f"Test Data Shape: {testX.shape}")

# #                 st.write("Go to the next page to build and train the model.")

# #         except Exception as e:
# #             st.write(f"Error loading data: {e}")
# #     else:
# #         st.write("Please enter a valid Google Drive file ID to load the dataset.")

# # # Page 2: Build and train model
# # def page_build_and_train():
# #     if 'trainX' not in st.session_state:
# #         st.warning("Please complete the data preprocessing step first.")
# #         return

# #     st.header("Step 3: Build and Train LSTM Model")
# #     num_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=5, value=2)
# #     lstm_units = [st.slider(f"LSTM Units for Layer {i + 1}", min_value=10, max_value=200, value=64) for i in range(num_layers)]
# #     dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
# #     activation_function = st.selectbox("Activation Function", ["relu", "tanh"])
# #     loss_function = st.selectbox("Loss Function", ["mse", "mae"])
# #     optimizer = st.selectbox("Optimizer", ["adam", "sgd"])
# #     epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15)
# #     batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=600)
# #     bidirectional = st.checkbox("Bidirectional LSTM")
# #     attention = st.checkbox("Add Attention Layer")

# #     if st.button("Train Model"):
# #         st.write("Starting model training...")  # Log message to indicate training has started
# #         try:
# #             model = Sequential()
# #             for i in range(num_layers):
# #                 if bidirectional:
# #                     model.add(Bidirectional(LSTM(lstm_units[i], activation=activation_function, input_shape=(st.session_state['sequence_length'], len(st.session_state['columns_to_scale'])), return_sequences=(i != num_layers - 1))))
# #                 else:
# #                     model.add(LSTM(lstm_units[i], activation=activation_function, input_shape=(st.session_state['sequence_length'], len(st.session_state['columns_to_scale'])), return_sequences=(i != num_layers - 1)))
# #                 if i != num_layers - 1:
# #                     model.add(Dropout(dropout_rate))

# #             if attention:
# #                 model.add(Attention())

# #             model.add(Dense(st.session_state['output_sequence_length']))
# #             model.compile(optimizer=optimizer, loss=loss_function)
# #             early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# #             epoch_time_callback = EpochTimeCallback()

# #             history = model.fit(st.session_state['trainX'], st.session_state['trainY'], epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, epoch_time_callback])

# #             st.session_state['model'] = model
# #             st.session_state['history'] = history.history
# #             st.write("Model Training Completed")

# #         except Exception as e:
# #             st.write(f"Error training model: {e}")

# # # Function to evaluate model
# # def page_evaluate_model():
# #     if 'model' not in st.session_state:
# #         st.warning("Please train the model first.")
# #         return

# #     st.header("Step 4: Evaluate Model")

# #     model = st.session_state['model']
# #     testX = st.session_state['testX']
# #     testY = st.session_state['testY']
# #     scaler = st.session_state['scaler']
# #     target_column = st.session_state['target_column']

# #     # Predict
# #     predicted = model.predict(testX)

# #     # Create a dummy array to match the expected shape for inverse transform
# #     dummy_array = np.zeros((predicted.shape[0], len(st.session_state['columns_to_scale'])))
# #     dummy_array[:, 0] = predicted[:, 0]  # Assuming target column is the first column

# #     # Inverse transform only the relevant column
# #     predicted_inverse = scaler.inverse_transform(dummy_array)[:, 0]
# #     actual_inverse = scaler.inverse_transform(testY)

# #     # Plot predictions vs actuals
# #     fig, ax = plt.subplots()
# #     ax.plot(actual_inverse[:100, 0], label='Actual')
# #     ax.plot(predicted_inverse[:100], label='Predicted')
# #     ax.set_title("Predicted vs Actual Values")
# #     ax.legend()

# #     st.pyplot(fig)

# #     st.write("Model evaluation completed. Predictions vs actuals plot displayed.")

# # # Main function to run the Streamlit app
# # def main():
# #     st.sidebar.title("LSTM Time Series Forecasting")
# #     page = st.sidebar.selectbox("Select a Page", ["Load Dataset and Preprocess Data", "Build and Train Model", "Evaluate Model"])

# #     if page == "Load Dataset and Preprocess Data":
# #         page_load_and_preprocess()
# #     elif page == "Build and Train Model":
# #         page_build_and_train()
# #     elif page == "Evaluate Model":
# #         page_evaluate_model()

# # if __name__ == "__main__":
# #     main()
# #-----------------------------------------------------------------actual code ----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Bidirectional, Attention
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import time
import os
import random
import gdown

# Custom callback to log epoch times
class EpochTimeCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        logs['epoch_time'] = epoch_time
        st.write(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")

# Function to create sequences
def create_sequences(data, sequence_length, output_sequence_length, target_column):
    X, y = [], []
    data_array = data.values
    total_length = sequence_length + output_sequence_length
    for i in range(len(data_array) - total_length + 1):
        seq_x = data_array[i:(i + sequence_length)]
        seq_y = data_array[i + sequence_length:(i + sequence_length + output_sequence_length), target_column]
        seq_y = seq_y.reshape(-1)
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to preprocess data
def preprocess_data(df, misisdn_column, event_column_date, columns_to_scale, target_column, sequence_length, output_sequence_length):
    msisdns = df[misisdn_column].unique()
    train_msisdns, test_msisdns = train_test_split(msisdns, test_size=0.2, random_state=42)
    train_data = df[df[misisdn_column].isin(train_msisdns)]
    test_data = df[df[misisdn_column].isin(test_msisdns)]
    scaler = RobustScaler()
    train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
    test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])

    X_train_dict, y_train_dict = {}, {}
    for msisdn, group in train_data.groupby(misisdn_column):
        group = group.sort_values(event_column_date)
        lstm_data = group[columns_to_scale]
        if len(lstm_data) >= sequence_length + output_sequence_length:
            X_train, y_train = create_sequences(lstm_data, sequence_length, output_sequence_length, df.columns.get_loc(target_column))
            X_train_dict[msisdn] = X_train
            y_train_dict[msisdn] = y_train

    trainX = np.concatenate(list(X_train_dict.values()))
    trainY = np.concatenate(list(y_train_dict.values()))

    X_test_dict, y_test_dict = {}, {}
    for msisdn, group in test_data.groupby(misisdn_column):
        group = group.sort_values(event_column_date)
        lstm_data = group[columns_to_scale]
        if len(lstm_data) >= sequence_length + output_sequence_length:
            X_test, y_test = create_sequences(lstm_data, sequence_length, output_sequence_length, df.columns.get_loc(target_column))
            X_test_dict[msisdn] = X_test
            y_test_dict[msisdn] = y_test

    testX = np.concatenate(list(X_test_dict.values()))
    testY = np.concatenate(list(y_test_dict.values()))

    return trainX, trainY, testX, testY, scaler

# Function to fetch and read the CSV data from Google Drive
def load_data_from_gdrive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'downloaded_data.csv'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Page 1: Load dataset and preprocess data
def page_load_and_preprocess():
    st.title("LSTM Time Series Forecasting")
    st.header("Step 1: Load Dataset")
    
    # Input field to enter the Google Drive file ID
    file_id = st.text_input("Enter the Google Drive file ID", placeholder="1dHebTpV8IlzXdmGebg53-K93RlhP-KWd")

    if file_id:
        try:
            df = load_data_from_gdrive(file_id)
            st.write("Data Preview:")
            st.write(df.head())

            # Proceed if data is loaded
            st.header("Step 2: Preprocess Data")
            misisdn_column = st.text_input("MSISDN Column", "Cust_Sub_Id")
            event_column_date = st.text_input("Date Column", "Event_Date")
            columns_to_scale = st.multiselect("Columns to Scale", df.columns.tolist(), default=['Total_Revenue', 'Data_Volume', 'OG_Call_Count'])
            df = df[[misisdn_column, event_column_date] + columns_to_scale]
            target_column = st.selectbox("Target Column", df.columns.tolist())
            sequence_length = st.number_input("Sequence Length", min_value=1, max_value=100, value=30)
            output_sequence_length = st.number_input("Output Sequence Length", min_value=1, max_value=10, value=1)

            if st.button("Preprocess Data"):
                st.write("Starting data preprocessing...")  # Log message for preprocessing
                trainX, trainY, testX, testY, scaler = preprocess_data(df, misisdn_column, event_column_date, columns_to_scale, target_column, sequence_length, output_sequence_length)
                st.session_state['trainX'] = trainX
                st.session_state['trainY'] = trainY
                st.session_state['testX'] = testX
                st.session_state['testY'] = testY
                st.session_state['scaler'] = scaler
                st.session_state['df'] = df
                st.session_state['columns_to_scale'] = columns_to_scale
                st.session_state['target_column'] = target_column
                st.session_state['sequence_length'] = sequence_length
                st.session_state['output_sequence_length'] = output_sequence_length
                st.session_state['misisdn_column'] = misisdn_column
                st.session_state['event_column_date'] = event_column_date

                st.write("Data Preprocessing Completed")
                st.write(f"Training Data Shape: {trainX.shape}")
                st.write(f"Test Data Shape: {testX.shape}")

                st.write("Go to the next page to build and train the model.")

        except Exception as e:
            st.write(f"Error loading data: {e}")
    else:
        st.write("Please enter a valid Google Drive file ID to load the dataset.")

# Page 2: Build and train model
def page_build_and_train():
    if 'trainX' not in st.session_state:
        st.warning("Please complete the data preprocessing step first.")
        return

    st.header("Step 3: Build and Train LSTM Model")
    num_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=5, value=2)
    lstm_units = [st.slider(f"LSTM Units for Layer {i + 1}", min_value=10, max_value=200, value=64) for i in range(num_layers)]
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
    activation_function = st.selectbox("Activation Function", ["relu", "tanh"])
    loss_function = st.selectbox("Loss Function", ["mse", "mae"])
    optimizer = st.selectbox("Optimizer", ["adam", "sgd"])
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=600)
    bidirectional = st.checkbox("Bidirectional LSTM")
    attention = st.checkbox("Add Attention Layer")

    if st.button("Train Model"):
        st.write("Starting model training...")  # Log message to indicate training has started
        try:
            model = Sequential()
            for i in range(num_layers):
                if bidirectional:
                    model.add(Bidirectional(LSTM(lstm_units[i], activation=activation_function, input_shape=(st.session_state['sequence_length'], len(st.session_state['columns_to_scale'])), return_sequences=(i != num_layers - 1))))
                else:
                    model.add(LSTM(lstm_units[i], activation=activation_function, input_shape=(st.session_state['sequence_length'], len(st.session_state['columns_to_scale'])), return_sequences=(i != num_layers - 1)))
                if i != num_layers - 1:
                    model.add(Dropout(dropout_rate))

            if attention:
                model.add(Attention())

            model.add(Dense(st.session_state['output_sequence_length']))
            model.compile(optimizer=optimizer, loss=loss_function)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            epoch_time_callback = EpochTimeCallback()

            # ModelCheckpoint callback to save model weights after each epoch
            checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')

            history = model.fit(st.session_state['trainX'], st.session_state['trainY'],
                                validation_data=(st.session_state['testX'], st.session_state['testY']),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping, epoch_time_callback, checkpoint_callback])

            st.write("Model Training Completed")

            # Plotting the training and validation loss
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.write(f"Error during model training: {e}")
            
# Page 3: Generate predictions
def page_generate_predictions():
    if 'model' not in st.session_state:
        st.warning("Please complete the model training step first.")
        return

    st.header("Step 4: Make Predictions")
    random_msisdns = st.multiselect("Select MSISDNs for Prediction",
                                    st.session_state['df'][st.session_state['misisdn_column']].unique(),
                                    default=random.sample(
                                        list(st.session_state['df'][st.session_state['misisdn_column']].unique()), 5))
    rolling_prediction = st.checkbox("Use Rolling Prediction")

    if st.button("Generate Predictions"):
        for msisdn in random_msisdns:
            group = st.session_state['df'][st.session_state['df'][st.session_state['misisdn_column']] == msisdn]
            group = group.sort_values(st.session_state['event_column_date'])
            lstm_data = group[st.session_state['columns_to_scale']]
            if len(lstm_data) >= st.session_state['sequence_length'] + st.session_state['output_sequence_length']:
                X_test, y_test = create_sequences(lstm_data, st.session_state['sequence_length'],
                                                  st.session_state['output_sequence_length'],
                                                  st.session_state['df'].columns.get_loc(
                                                      st.session_state['target_column']))

                if rolling_prediction:
                    # Initialize lists to store predictions and actual values
                    predictions_msisdn = []
                    actuals_msisdn = []

                    # Initialize the input sequence (the first 30 days)
                    input_sequence = X_test[0]

                    for i in range(len(y_test)):
                        # Predict the next day
                        prediction = st.session_state['model'].predict(
                            input_sequence.reshape(1, st.session_state['sequence_length'],
                                                   len(st.session_state['columns_to_scale'])))

                        # Store the prediction and actual value
                        predictions_msisdn.append(prediction[0, 0])
                        actuals_msisdn.append(y_test[i])

                        # Update the input sequence
                        input_sequence = np.roll(input_sequence, -1, axis=0)
                        input_sequence[-1] = prediction

                    # Convert lists to numpy arrays
                    predictions_msisdn = np.array(predictions_msisdn)
                    actuals_msisdn = np.array(actuals_msisdn)

                    # Add dummy columns to match the shape expected by the scaler
                    dummy_columns = np.zeros(
                        (predictions_msisdn.shape[0], len(st.session_state['columns_to_scale']) - 1))
                    predictions_msisdn_full = np.hstack([predictions_msisdn.reshape(-1, 1), dummy_columns])
                    actuals_msisdn_full = np.hstack([actuals_msisdn.reshape(-1, 1), dummy_columns])

                    # Inverse transform predictions and actuals
                    predictions_msisdn_inverse = st.session_state['scaler'].inverse_transform(predictions_msisdn_full)[
                                                 :, 0]
                    actuals_msisdn_inverse = st.session_state['scaler'].inverse_transform(actuals_msisdn_full)[:, 0]

                    # Plotting
                    plt.figure(figsize=(10, 5))
                    plt.plot(group[st.session_state['event_column_date']].iloc[-len(predictions_msisdn):],
                             predictions_msisdn_inverse, label='Predicted')
                    plt.plot(group[st.session_state['event_column_date']].iloc[-len(actuals_msisdn):],
                             actuals_msisdn_inverse, label='Actual')
                    plt.title(f"Rolling Predictions for MSISDN: {msisdn}")
                    plt.xlabel("Date")
                    plt.ylabel("Total Revenue")
                    plt.legend()
                    st.pyplot(plt)

                else:
                    predictions = st.session_state['model'].predict(X_test)
                    predictions = st.session_state['scaler'].inverse_transform(
                        np.hstack([predictions,
                                   np.zeros((predictions.shape[0], len(st.session_state['columns_to_scale']) - 1))]))[:,
                                  0]
                    actuals = st.session_state['scaler'].inverse_transform(
                        np.hstack(
                            [y_test, np.zeros((y_test.shape[0], len(st.session_state['columns_to_scale']) - 1))]))[:, 0]

                    plt.figure(figsize=(10, 5))
                    plt.plot(group[st.session_state['event_column_date']].iloc[-len(predictions):], predictions,
                             label='Predicted')
                    plt.plot(group[st.session_state['event_column_date']].iloc[-len(actuals):], actuals, label='Actual')
                    plt.title(f"Predictions for MSISDN: {msisdn}")
                    plt.xlabel("Date")
                    plt.ylabel("Total Revenue")
                    plt.legend()
                    st.pyplot(plt)
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Load and Preprocess Data", "Build and Train Model", "Generate Predictions"])

if page == "Load and Preprocess Data":
    page_load_and_preprocess()
elif page == "Build and Train Model":
    page_build_and_train()
elif page == "Generate Predictions":
    page_generate_predictions()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

