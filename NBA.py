import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
df1 = pd.read_excel(r'C:\Kampus\Semester 5\Data Mining\Data_NBA(2).xlsx')
df2 = pd.read_excel(r'C:\Kampus\Semester 5\Data Mining\NBA(Salary).xlsx')
merged_df = pd.merge(df1, df2, on='Player', how='inner')


st.title("NBA Player Analysis")

# Sidebar for navigation
option = st.sidebar.selectbox(
    'Select Analysis',
    ('PTS vs Salary', 'eFG% vs Salary', 'TRB vs Salary', '3P% vs Salary', 'AST - TOV vs Salary', 'BLK + STL vs Salary')
)


if option == 'PTS vs Salary':
    st.header('Points (PTS) vs. Salary')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Salary', y='PTS', data=merged_df, ax=ax)
    for i, row in merged_df.iterrows():
        ax.annotate(i, (row['Salary'], row['PTS']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Player's Salary")
    ax.set_ylabel("Player's PTS")
    ax.set_title("Player's Salary vs. Player's PTS")
    ax.grid(True)
    st.pyplot(fig)
    st.write("Consider to Buy (Low Salary, High PTS)")
    st.write(merged_df.iloc[11]) #Example player
    st.write("Avoid to Buy (High Salary, Low PTS)")
    st.write(merged_df.iloc[257]) #Example player


elif option == 'eFG% vs Salary':
  st.header('Effective Field Goal Percentage (eFG%) vs. Salary')
  # ... (similar code as above, but for eFG% vs. Salary)
  filtered_df = merged_df[merged_df['FGA'] >= 10]
  sorted_eFG_df = filtered_df.sort_values(by=['eFG%', 'FGA'], ascending=[False, True])
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.regplot(x='Salary', y='eFG%', data=sorted_eFG_df, ax=ax)
  for i, row in sorted_eFG_df.iterrows():
      ax.annotate(i, (row['Salary'], row['eFG%']), textcoords="offset points", xytext=(0,5), ha='center')
  ax.set_xlabel("Player's Salary")
  ax.set_ylabel("Player's eFG%")
  ax.set_title("Player's Salary vs. Player's eFG% (FGA >= 10)")
  ax.grid(True)
  st.pyplot(fig)
  st.write("Consider to Buy (Low Salary, High eFG%)")
  st.write(merged_df.iloc[56])
  st.write("Avoid to Buy (High Salary, Low eFG%)")
  st.write(merged_df.iloc[66])


# ... (add similar elif blocks for other analysis options)

elif option == 'TRB vs Salary':
    st.header('Total Rebounds (TRB) vs Salary')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Salary', y='TRB', data=merged_df, ax=ax)
    for i, row in merged_df.iterrows():
        ax.annotate(i, (row['Salary'], row['TRB']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Player's Salary")
    ax.set_ylabel("Player's TRB")
    ax.set_title("Player's Salary vs. Player's TRB")
    ax.grid(True)
    st.pyplot(fig)
    st.write("Consider to Buy (Low Salary, High TRB)")
    st.write(merged_df.iloc[97])
    st.write("Avoid to Buy (High Salary, Low TRB)")
    st.write(merged_df.iloc[410])

elif option == '3P% vs Salary':
    st.header('3-Point Percentage (3P%) vs Salary')
    fig, ax = plt.subplots(figsize=(10, 6))
    # Filter for players with minimum G >= 40 and 3PA >= 4
    filtered_df = merged_df[(merged_df['G'] >= 30) & (merged_df['3PA'] >= 4)]
    sns.regplot(x='Salary', y='3P%', data=filtered_df, ax=ax)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['3P%']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Player's Salary")
    ax.set_ylabel("Player's 3P%")
    ax.set_title("Player's Salary vs. Player's 3P% (G >= 30 and 3PA >= 4)")
    ax.grid(True)
    st.pyplot(fig)
    st.write("Consider to Buy (Low Salary, High 3P%)")
    st.write(merged_df.iloc[101])
    st.write("Avoid to Buy (High Salary, Low 3P%)")
    st.write(merged_df.iloc[223])


elif option == 'AST - TOV vs Salary':
    st.header('Assists (AST) - Turnovers (TOV) vs Salary')
    fig, ax = plt.subplots(figsize=(10, 6))
    merged_df['AST_minus_TOV'] = merged_df['AST'] - merged_df['TOV']
    filtered_df = merged_df[merged_df['G'] >= 40]
    sns.regplot(x='Salary', y='AST_minus_TOV', data=filtered_df, ax=ax)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['AST_minus_TOV']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Player's Salary")
    ax.set_ylabel("Player's AST_minus_TOV")
    ax.set_title("Player's Salary vs. Player's AST_minus_TOV (G >= 30 and 3PA >= 4)")
    ax.grid(True)
    st.pyplot(fig)
    st.write("Consider to Buy (Low Salary, High AST_minus_TOV)")
    st.write(merged_df.iloc[44])
    st.write("Avoid to Buy (High Salary, Low AST_minus_TOV)")
    st.write(merged_df.iloc[92])

elif option == 'BLK + STL vs Salary':
    st.header('Blocks (BLK) + Steals (STL) vs Salary')
    fig, ax = plt.subplots(figsize=(10, 6))
    merged_df['BLK_plus_STL'] = merged_df['BLK'] + merged_df['STL']
    filtered_df = merged_df[merged_df['G'] >= 40]
    sns.regplot(x='Salary', y='BLK_plus_STL', data=filtered_df, ax=ax)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['BLK_plus_STL']), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Player's Salary")
    ax.set_ylabel("Player's BLK_plus_STL")
    ax.set_title("Player's Salary vs. Player's BLK_plus_STL (G >= 30 and 3PA >= 4)")
    ax.grid(True)
    st.pyplot(fig)
    st.write("Consider to Buy (Low Salary, High BLK_plus_STL)")
    st.write(merged_df.iloc[191])
    st.write("Avoid to Buy (High Salary, Low BLK_plus_STL)")
    st.write(merged_df.iloc[272])
