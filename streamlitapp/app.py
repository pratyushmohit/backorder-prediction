import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.proto.Markdown_pb2 import Markdown
from inference import predict
import streamlit.components.v1 as components

st.title('Prediction of Backorders in Inventory Management')
st.header('A Random Forest model trained with a balanced subsample class weight')
st.markdown('created by: **Pratyush Mohit**')
st.subheader("Upload a csv file")

uploaded_file = st.file_uploader("Choose a file...", type=['csv'])
if uploaded_file is not None:
   dataframe = pd.read_csv(uploaded_file)
   st.write("Loading...displaying first five rows")
   st.dataframe(data=dataframe.head(), width=730, height=200)
   st.write("Predicting...")
   x = dataframe.drop('went_on_backorder', axis=1)
   fig, ax = plt.subplots()
   sns.heatmap(x.corr(), ax=ax)
   plt.title('Correlation Matrix')
   st.write(fig)
   predictions = predict(x)
   st.write(predictions)
   st.write('Done')

   