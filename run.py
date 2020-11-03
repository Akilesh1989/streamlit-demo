import streamlit as st
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import base64
from SessionState import _get_state


st.beta_set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def conjuction(*conditions):
    return functools.reduce(np.logical_and, conditions)


def get_default_options(df, column_name):
    return df[column_name].unique().tolist()


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

df = pd.concat(map(pd.read_csv, ['train.csv', 'test.csv']))
df = df.fillna(0)
df['Survived'] = df['Survived'].astype(int)

st.header("EXPLORING THE TITANIC DATASET")
sidebar_option = st.sidebar.beta_expander('FILTERS')

state = _get_state()

if state.sex is None:
    state.sex = ['Male', 'Female']
if sidebar_option.button('Fill all'):
    sex = sidebar_option.multiselect("Sex ", ('Male', 'Female'), default=['Male', 'Female'])
else:
    sex = sidebar_option.multiselect("Sex", ('Male', 'Female'), default=state.sex)
state.sex = sex
csex = df.Sex.isin([value.lower() for value in sex])
sidebar_option.markdown(" *** ")

if state.pclass is None:
    state.pclass = [1, 2, 3]
if sidebar_option.button('Fill all '):
    pclass = sidebar_option.multiselect("Pclass", (1, 2, 3), default=[1, 2, 3])
else:
    pclass = sidebar_option.multiselect("Pclass", (1, 2, 3), default=state.pclass)
state.pclass = pclass
cpclass = df.Pclass.isin(pclass)
sidebar_option.markdown(" *** ")

# Age filter
min_age = 0
max_age = 80
age = sidebar_option.slider('Min age', int(min(df['Age'])), int(max(df['Age'])) - 10, value=(min_age, max_age), step=10)
cmin_age = df.Age >= age[0]
cmax_age = df.Age <= age[1]
sidebar_option.markdown(" *** ")


# Embarked filter
embarked_unique_values = get_default_options(df, 'Embarked')
if state.embarked is None:
    state.embarked = embarked_unique_values
if sidebar_option.button("Fill all  "):
    embarked = sidebar_option.multiselect("Embarked", embarked_unique_values, default=embarked_unique_values)
else:
    embarked = sidebar_option.multiselect("Embarked", embarked_unique_values, default=state.embarked)
state.embarked = embarked
c_embarked = df.Embarked.isin(embarked)
sidebar_option.markdown(" *** ")

state.sync()

filtered_df = df[conjuction(csex, cpclass, cmin_age, cmax_age, c_embarked)]

if filtered_df.empty:
    st.write('INFO: Empty Dataset. Try modifying the values.')
else:

    container_1_show = st.beta_expander("Show table")
    with container_1_show:
        container_1 = st.beta_container()
        container_1.subheader('Looking at the data')
        container_1.write(filtered_df)
        container_1.markdown(get_table_download_link(filtered_df), unsafe_allow_html=True)
    
    container_2_show = st.beta_expander("Show basic count plots")
    with container_2_show:
        container_2 = st.beta_container()
        col1, col2, col3 = container_2.beta_columns([3, 3, 3])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Survived')
        sns.countplot(data=filtered_df, x='Survived')
        col1.pyplot(plt)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Gender')
        sns.countplot(x='Sex', data=filtered_df)
        col2.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Embarked')
        sns.countplot(x='Embarked', data=filtered_df)
        col3.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Pclass')
        sns.countplot(x='Pclass', data=filtered_df)
        col1.pyplot(fig)

    container_3_show = st.beta_expander("Show comparison plots")
    with container_3_show:
        container_2 = st.beta_container()
        col1, col2, col3 = container_2.beta_columns([3, 3, 3])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Gender')
        sns.countplot(x='Survived', hue='Sex', data=filtered_df)
        col1.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Pclass')
        sns.countplot(x='Survived', hue='Pclass', data=filtered_df)
        col2.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Embarked')
        sns.countplot(x='Survived', hue='Embarked', data=filtered_df)
        col3.pyplot(fig)

    
    