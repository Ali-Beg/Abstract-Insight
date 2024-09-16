# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.data import Dataset
# from tensorflow.keras.models import load_model as tf_load_model  # Renaming to avoid conflict

# AUTOTUNE = tf.data.AUTOTUNE

# # Load the model
# @st.cache_resource
# def load_my_model():
#     model = tf_load_model('model/skimlit_model_large.keras')  # Update with the correct path to your model
#     return model

# # Define a character splitting function
# def split_chars(text):
#     return " ".join(list(text))

# # Prediction function
# def prep_abstract_and_predict(model, abstract_text):
#     """Takes in the raw abstract text and returns a DataFrame with predictions."""
#     abstract_sentences = abstract_text.split('. ')
#     abstract_chars = [split_chars(sentence) for sentence in abstract_sentences]
#     abstract_line_numbers = np.array(range(len(abstract_sentences)))
#     abstract_total_lines = np.ones(len(abstract_line_numbers)) * len(abstract_line_numbers)

#     # Dummy labels (for dataset creation, not used in prediction)
#     dummy_labels = tf.one_hot(np.random.randint(low=0, high=5, size=len(abstract_line_numbers)), depth=5)

#     # Create the dataset
#     abstract_dataset = Dataset.zip(
#         Dataset.from_tensor_slices((
#             tf.one_hot(abstract_line_numbers, depth=15),
#             tf.one_hot(abstract_total_lines, depth=20),
#             abstract_sentences,
#             abstract_chars
#         )),
#         Dataset.from_tensor_slices(dummy_labels)
#     ).batch(32).prefetch(AUTOTUNE)

#     # Get predictions
#     pred_probs = model.predict(abstract_dataset)
#     preds = tf.argmax(pred_probs, axis=1)

#     # Create a DataFrame for the results (showing only predictions and probabilities)
#     abstract_df = pd.DataFrame({
#         'text': abstract_sentences,
#         'prediction': [(
#             'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'BACKGROUND'
#         )[pred] for pred in preds],
#         'probability': tf.reduce_max(pred_probs, axis=1).numpy()  # Convert tensor to numpy for readability
#     })

#     return abstract_df

# # Streamlit app UI
# st.title("📝 Medical Abstract Classification")

# st.write("""
#     Input a medical abstract below, and our model will classify each line into:
#     **OBJECTIVE**, **METHODS**, **RESULTS**, **CONCLUSIONS**, or **BACKGROUND**.
# """)

# # Input for abstract text
# abstract_text = st.text_area("📄 Enter the medical abstract here:", height=250)

# # Button to trigger classification
# if st.button("🔍 Classify Abstract"):
#     if abstract_text:
#         # Load the model
#         model = load_my_model()
        
#         # Get the predictions
#         results_df = prep_abstract_and_predict(model, abstract_text)
        
#         # Group the results by the prediction type
#         grouped_results = results_df.groupby('prediction')

#         # Display the results grouped by label
#         if 'OBJECTIVE' in grouped_results.groups:
#             st.write("### 🎯 Objective")
#             for index, row in grouped_results.get_group('OBJECTIVE').iterrows():
#                 st.markdown(f"**{row['text']}** - Confidence: `{row['probability']:.2f}`")
        
#         if 'METHODS' in grouped_results.groups:
#             st.write("### 🛠️ Methods")
#             for index, row in grouped_results.get_group('METHODS').iterrows():
#                 st.markdown(f"**{row['text']}** - Confidence: `{row['probability']:.2f}`")

#         if 'RESULTS' in grouped_results.groups:
#             st.write("### 📊 Results")
#             for index, row in grouped_results.get_group('RESULTS').iterrows():
#                 st.markdown(f"**{row['text']}** - Confidence: `{row['probability']:.2f}`")

#         if 'CONCLUSIONS' in grouped_results.groups:
#             st.write("### 📝 Conclusions")
#             for index, row in grouped_results.get_group('CONCLUSIONS').iterrows():
#                 st.markdown(f"**{row['text']}** - Confidence: `{row['probability']:.2f}`")

#         if 'BACKGROUND' in grouped_results.groups:
#             st.write("### 🏛️ Background")
#             for index, row in grouped_results.get_group('BACKGROUND').iterrows():
#                 st.markdown(f"**{row['text']}** - Confidence: `{row['probability']:.2f}`")
#     else:
#         st.warning("⚠️ Please enter a medical abstract.")


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import load_model as tf_load_model  # Renaming to avoid conflict

AUTOTUNE = tf.data.AUTOTUNE

# Load the model
@st.cache_resource
def load_my_model():
    model = tf_load_model('model/skimlit_model_large.keras')  # Update with the correct path to your model
    return model

# Define a character splitting function
def split_chars(text):
    return " ".join(list(text))

# Prediction function
def prep_abstract_and_predict(model, abstract_text):
    """Takes in the raw abstract text and returns a DataFrame with predictions."""
    abstract_sentences = abstract_text.split('. ')
    abstract_chars = [split_chars(sentence) for sentence in abstract_sentences]
    abstract_line_numbers = np.array(range(len(abstract_sentences)))
    abstract_total_lines = np.ones(len(abstract_line_numbers)) * len(abstract_line_numbers)

    # Dummy labels (for dataset creation, not used in prediction)
    dummy_labels = tf.one_hot(np.random.randint(low=0, high=5, size=len(abstract_line_numbers)), depth=5)

    # Create the dataset
    abstract_dataset = Dataset.zip(
        Dataset.from_tensor_slices((
            tf.one_hot(abstract_line_numbers, depth=15),
            tf.one_hot(abstract_total_lines, depth=20),
            abstract_sentences,
            abstract_chars
        )),
        Dataset.from_tensor_slices(dummy_labels)
    ).batch(32).prefetch(AUTOTUNE)

    # Get predictions
    pred_probs = model.predict(abstract_dataset)
    preds = tf.argmax(pred_probs, axis=1)

    # Create a DataFrame for the results (showing only predictions)
    abstract_df = pd.DataFrame({
        'text': abstract_sentences,
        'prediction': [(
            'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'BACKGROUND'
        )[pred] for pred in preds]
    })

    return abstract_df

# Streamlit app UI
st.title("📝 Medical Abstract Classification")

st.write("""
    Input a medical abstract below, and our model will classify each line into:
    **OBJECTIVE**, **METHODS**, **RESULTS**, **CONCLUSIONS**, or **BACKGROUND**.
""")

# Input for abstract text
abstract_text = st.text_area("📄 Enter the medical abstract here:", height=250)

# Button to trigger classification
if st.button("🔍 Classify Abstract"):
    if abstract_text:
        # Load the model
        model = load_my_model()
        
        # Get the predictions
        results_df = prep_abstract_and_predict(model, abstract_text)
        
        # Group the results by the prediction type
        grouped_results = results_df.groupby('prediction')

        # Display the results grouped by label
        if 'OBJECTIVE' in grouped_results.groups:
            st.write("### 🎯 Objective")
            for index, row in grouped_results.get_group('OBJECTIVE').iterrows():
                st.markdown(f"**{row['text']}**")
        
        if 'METHODS' in grouped_results.groups:
            st.write("### 🛠️ Methods")
            for index, row in grouped_results.get_group('METHODS').iterrows():
                st.markdown(f"**{row['text']}**")

        if 'RESULTS' in grouped_results.groups:
            st.write("### 📊 Results")
            for index, row in grouped_results.get_group('RESULTS').iterrows():
                st.markdown(f"**{row['text']}**")

        if 'CONCLUSIONS' in grouped_results.groups:
            st.write("### 📝 Conclusions")
            for index, row in grouped_results.get_group('CONCLUSIONS').iterrows():
                st.markdown(f"**{row['text']}**")

        if 'BACKGROUND' in grouped_results.groups:
            st.write("### 🏛️ Background")
            for index, row in grouped_results.get_group('BACKGROUND').iterrows():
                st.markdown(f"**{row['text']}**")
    else:
        st.warning("⚠️ Please enter a medical abstract.")
