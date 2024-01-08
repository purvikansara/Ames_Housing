# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px
import plotly.io as pio
pio.templates.default='seaborn'

# Add title
st.title("EDA with Streamlit")

# Define the columns you want to use 
columns_to_use = ['SalePrice', 'Living Area Sqft', 'Lot Frontage', 'Bldg Type', 'Bedroom',
                    'Total Full Baths','MS Zoning','Street', 
                    'Alley','Utilities']
# Function for loading data
# Adding data caching
@st.cache_data
def load_data():
    fpath =  "Data/ames-housing-dojo-for-ml.csv"
    df = pd.read_csv(fpath)
    df = df.set_index("PID")
    df = df[columns_to_use]
    return df

# load the data 
df = load_data()


##########################################################################################################################
# Solution to Plotly Practice Assignment

st.markdown("#### Explore Features vs. Sale Price with Plotly")
# Add a selectbox for all possible features (exclude SalePrice)
# Copy list of columns
features_to_use = columns_to_use[:]
# Define target
target = 'SalePrice'
# Remove target from list of features
features_to_use.remove(target)

# Add a selectbox for all possible columns
feature = st.selectbox(label="Select a feature to compare with Sale Price", options=features_to_use)



def plotly_numeric_vs_target(df, x, y='SalePrice', trendline='ols',add_hoverdata=True):
    if add_hoverdata == True:
        hover_data = list(df.columns)
    else: 
        hover_data = None
        
    pfig = px.scatter(df, x=x, y=y,width=800, height=600,
                     hover_data=hover_data,
                      trendline=trendline,
                      trendline_color_override='red',
                     title=f"{x} vs. {y}")
    
    pfig.update_traces(marker=dict(size=3),
                      line=dict(dash='dash'))
    return pfig

def plotly_categorical_vs_target(df, x, y='SalePrice', histfunc='avg', width=800,height=500):
    fig = px.histogram(df, x=x,y=y, color=x, width=width, height=height,
                       histfunc=histfunc, title=f'Compare {histfunc.title()} {y} by {x}')
    fig.update_layout(showlegend=False)
    return fig

# Conditional statement to determine which function to use
if df[feature].dtype == 'object':
    fig_vs  = plotly_categorical_vs_target(df, x = feature)
else:
    fig_vs  = plotly_numeric_vs_target(df, x = feature)

st.plotly_chart(fig_vs)
