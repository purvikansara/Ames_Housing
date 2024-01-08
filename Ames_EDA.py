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
# Display an interactive dataframe
st.header("Displaying a DataFrame")
st.dataframe(df, width=800)

# Display descriptive statistics
st.markdown('#### Descriptive Statistics')
st.dataframe(df.describe().round(2))

# Display .info()
# In order to display output on our app, we first need to capture it. 
# We can use an IO buffer to capture the output, then we will use the getvalue() argument to retrive it
from io import StringIO
# Create a string buffer to capture the content
buffer = StringIO()
# Write the info into the buffer
df.info(buf=buffer)
# Retrieve the content from the buffer
summary_info = buffer.getvalue()

# What the output looks like in the console
st.markdown("#### Summary Info")
st.text(summary_info)


# We could display the output series as a dataframe
st.markdown("#### Null Values as dataframe")
nulls =df.isna().sum()
st.dataframe(nulls)
# Create a string buffer to capture the content
buffer = StringIO()
# Write the content into the buffer...use to_string
df.isna().sum().to_string(buffer)
# Retrieve the content from the buffer
null_values = buffer.getvalue()
# Use Streamlit to display the info
st.markdown("#### Null Values as String")
st.text(null_values)


# Function for returning a histogram fig
def plot_hist(df, column):
    fig, ax = plt.subplots()
    sns.histplot(data = df, x = column)
    return fig 
# Define the fig object with the function    
fig = plot_hist(df, 'SalePrice')
st.markdown("#### Display a plt plot")
st.pyplot(fig)

def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
 
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
    
  return fig, ax
# Define returned variables
fig, ax = explore_categorical(df, 'Alley')
st.markdown("#### Displaying a plot from explore_categorical function")
st.pyplot(fig)


def explore_numeric(df, x, figsize=(6,5) ):
  """Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  # Ensure plot is shown before message
  plt.show()
  return fig
st.markdown("#### Displaying a plot from explore_numeric function")
fig = explore_numeric(df, 'Lot Frontage')
st.pyplot(fig)

# Add a selectbox for all possible features
column = st.selectbox(label="Select a column", options=columns_to_use)
# Conditional statement to determine which function to use
if df[column].dtype == 'object':
    fig, ax  = explore_categorical(df, column)
else:
    fig = explore_numeric(df, column)
    
st.markdown("#### Displaying appropriate plot based on selected column")
# Display appropriate eda plots
st.pyplot(fig)

# Use plotly for explore functions
def plotly_explore_numeric(df, x):
    fig = px.histogram(df,x=x,marginal='box',title=f'Distribution of {x}', 
                      width=1000, height=500)
    return fig
def plotly_explore_categorical(df, x):
    fig = px.histogram(df,x=x,color=x,title=f'Distribution of {x}', 
                          width=1000, height=500)
    fig.update_layout(showlegend=False)
    return fig
# Conditional statement to determine which function to use
if df[column].dtype == 'object':
    fig = plotly_explore_categorical(df, column)
else:
    fig = plotly_explore_numeric(df, column)
    
st.markdown("#### Displaying appropriate Plotly plot based on selected column")
# Display appropriate eda plots
st.plotly_chart(fig)


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

