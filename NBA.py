import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df1 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/Data_NBA(2).xlsx')
    df2 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/NBA(Salary).xlsx')

    merged_df = pd.merge(df1, df2, on='Player', how='inner')
    merged_df['AST - TOV'] = merged_df['AST'] - merged_df['TOV']
    merged_df['BLK + STL'] = merged_df['BLK'] + merged_df['STL']
    features = ['PTS', 'eFG%', 'TRB', 'AST - TOV', 'BLK + STL', '3P%']
    X = merged_df[features]

    # Threshold
    thresholds = {stat: X[stat].mean() for stat in features}
    merged_df['Quality Player'] = (
        (merged_df['PTS'] > thresholds['PTS']) &
        (merged_df['TRB'] > thresholds['TRB']) &
        (merged_df['AST - TOV'] > thresholds['AST - TOV']) &
        (merged_df['BLK + STL'] > thresholds['BLK + STL']) &
        (merged_df['eFG%'] > thresholds['eFG%'])
    ).astype(int)

    salary_threshold = merged_df['Salary'].mean()
    merged_df['Salary Player'] = (merged_df['Salary'] > salary_threshold).astype(int)

    # Random Forest
    y = merged_df['Quality Player']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

    return merged_df, rf_model

merged_df, rf_model = load_data()

st.title("NBA Player Analysis and Classication")

st.sidebar.header("Analysis Options")
option = st.sidebar.selectbox(
    "Select Analysis",
    [
        "PTS vs Salary",
        "eFG% vs Salary",
        "TRB vs Salary",
        "3P% vs Salary",
        "AST - TOV vs Salary",
        "BLK + STL vs Salary",
        "Quality Players",
        "Quality Player (KNN)",
        "Classify Quality Player",
        "Random Forest Accuracy", 
        "KNN Accuracy",
    ],
)


st.sidebar.header("Filters")
games_played = st.sidebar.slider("Minimum Games Played", 0, 82, 0)
salary_range = st.sidebar.slider(
    "Salary Range (in millions)", 
    int(merged_df['Salary'].min() / 1e6), 
    int(merged_df['Salary'].max() / 1e6), 
    (0, 51)
)

st.sidebar.header("Player Data by Index")
player_index = st.sidebar.number_input(
    "Enter Player Index", 
    min_value=0, 
    max_value=len(merged_df) - 1, 
    value=0, 
    step=1
)

selected_player = merged_df.iloc[player_index]
st.sidebar.write("Player Data:")
st.sidebar.write(selected_player)

filtered_df = merged_df[
    (merged_df['G'] >= games_played) & 
    (merged_df['Salary'] / 1e6 >= salary_range[0]) & 
    (merged_df['Salary'] / 1e6 <= salary_range[1])
]

if option == "Quality Players":
    st.header("Quality Player Analysis")
    quality_players = filtered_df[filtered_df['Quality Player'] == 1]
    st.write(f"Number of Quality Players: {len(quality_players)}")
    st.dataframe(quality_players[['Player', 'PTS', 'TRB', 'AST', 'BLK', 'STL', 'Quality Player', 'Salary Player']])

elif option == "Quality Player (KNN)":
    st.header("Classify Quality Player Status (KNN)")
    player_stats = {
        'PTS': st.number_input("Enter Points (PTS)", min_value=0, value=0),
        'TRB': st.number_input("Enter Total Rebounds (TRB)", min_value=0, value=0),
        'AST': st.number_input("Enter Assists (AST)", min_value=0, value=0),
        'TOV': st.number_input("Enter Turnovers (TOV)", min_value=0, value=0),
        'BLK': st.number_input("Enter Blocks (BLK)", min_value=0, value=0),
        'STL': st.number_input("Enter Steals (STL)", min_value=0, value=0),
        'eFG%': st.number_input("Enter Effective Field Goal Percentage (eFG%)", min_value=0.0, value=0.00),
        '3P%': st.number_input("Enter 3-Point Percentage (3P%)", min_value=0.0, value=0.00),
    }
    ast_tov = player_stats['AST'] - player_stats['TOV']
    blk_stl = player_stats['BLK'] + player_stats['STL']
    input_data = np.array([
        player_stats['PTS'],  
        player_stats['eFG%'],
        player_stats['TRB'], 
        ast_tov, 
        blk_stl, 
        player_stats['3P%']
    ]).reshape(1, -1)
    classification = knn_model.predict(input_data)

    if classification == 1:
        st.success("This player is a Quality Player (KNN)!!!")
    else:
        st.warning("This player is not a Quality Player (KNN).")

elif option == "PTS vs Salary":
    st.header("Points (PTS) vs. Salary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Salary', y='PTS', hue='Quality Player', palette='viridis', ax=ax)
    sns.regplot(data=filtered_df, x='Salary', y='PTS', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression"})
    ax.legend()
    ax.set_title("Points (PTS) vs Salary")
    ax.grid(True)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['PTS']), fontsize=8, alpha=0.7)
    st.pyplot(fig)

elif option == "eFG% vs Salary":
    st.header("eFG% vs. Salary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Salary', y='eFG%', hue='Quality Player', palette='viridis', ax=ax)
    sns.regplot(data=filtered_df, x='Salary', y='eFG%', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression"})
    ax.legend()
    ax.set_title("eFG% vs Salary")
    ax.grid(True)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['eFG%']), fontsize=8, alpha=0.7)
    st.pyplot(fig)

elif option == "3P% vs Salary":
    st.header("3P% vs. Salary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Salary', y='3P%', hue='Quality Player', palette='viridis', ax=ax)
    sns.regplot(data=filtered_df, x='Salary', y='3P%', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression"})
    ax.legend()
    ax.set_title("3P% vs Salary")
    ax.grid(True)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['3P%']), fontsize=8, alpha=0.7)
    st.pyplot(fig)

elif option == "AST - TOV vs Salary":
    st.header("Assists (AST) - Turnovers (TOV) vs. Salary")
    filtered_df['AST - TOV'] = filtered_df['AST'] - filtered_df['TOV']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Salary', y='AST - TOV', hue='Quality Player', palette='viridis', ax=ax)
    sns.regplot(data=filtered_df, x='Salary', y='AST - TOV', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression"})
    ax.legend()
    ax.set_title("AST - TOV vs Salary")
    ax.grid(True)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['AST - TOV']), fontsize=8, alpha=0.7)
    st.pyplot(fig)

elif option == "BLK + STL vs Salary":
    st.header("Blocks (BLK) + Steals (STL) vs. Salary")
    filtered_df['BLK + STL'] = filtered_df['BLK'] + filtered_df['STL']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='Salary', y='BLK + STL', hue='Quality Player', palette='viridis', ax=ax)
    sns.regplot(data=filtered_df, x='Salary', y='BLK + STL', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression"})
    ax.legend()
    ax.set_title("BLK + STL vs Salary")
    ax.grid(True)
    for i, row in filtered_df.iterrows():
        ax.annotate(i, (row['Salary'], row['BLK + STL']), fontsize=8, alpha=0.7)
    st.pyplot(fig)
    st.pyplot(fig)

elif option == "Classify Quality Player":
    st.header("Classify Quality Player Status")
    player_stats = {
        'PTS': st.number_input("Enter Points (PTS)", min_value=0, value=0),
        'TRB': st.number_input("Enter Total Rebounds (TRB)", min_value=0, value=0),
        'AST': st.number_input("Enter Assists (AST)", min_value=0, value=0),
        'TOV': st.number_input("Enter Turnovers (TOV)", min_value=0, value=0),
        'BLK': st.number_input("Enter Blocks (BLK)", min_value=0, value=0),
        'STL': st.number_input("Enter Steals (STL)", min_value=0, value=0),
        'eFG%': st.number_input("Enter Effective Field Goal Percentage (eFG%)", min_value=0.0, value=0.00),
        '3P%': st.number_input("Enter 3-Point Percentage (3P%)", min_value=0.0, value=0.00),
    }
    
elif option == "Random Forest Accuracy":
    st.header("Random Forest Model Accuracy")
    
    # Hitung akurasi model
    y = merged_df['Quality Player']
    features = ['PTS', 'eFG%', 'TRB', 'AST - TOV', 'BLK + STL', '3P%']
    X = merged_df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Tampilkan hasil akurasi
    st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")
    
    # Tambahkan visualisasi jika diperlukan
    st.subheader("Feature Importance")
    feature_importance = rf_model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=features, y=feature_importance, ax=ax)
    ax.set_title("Feature Importance in Random Forest Model")
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Features")
    st.pyplot(fig)

    ast_tov = player_stats['AST'] - player_stats['TOV']
    blk_stl = player_stats['BLK'] + player_stats['STL']
    input_data = np.array([
        player_stats['PTS'],  
        player_stats['TRB'], 
        ast_tov, 
        blk_stl, 
        player_stats['eFG%'],
        player_stats['3P%']
    ]).reshape(1, -1)
    classification = rf_model.predict(input_data)

    if classification == 1:
        st.success("This player is a Quality Player!!!")
    else:
        st.warning("This player is not a Quality Player.")

elif option == "KNN Accuracy":
    st.header("KNN Model Accuracy")
    st.write(f"Model Accuracy: **{knn_accuracy * 100:.2f}%**")

    # Visualisasi feature importance tidak relevan untuk KNN, jadi hanya tampilkan data
    st.subheader("Note")
    st.write("KNN does not provide feature importance directly, as it is a distance-based algorithm.")

