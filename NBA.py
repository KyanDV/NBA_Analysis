import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    df1 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/Data_NBA(2).xlsx')
    df2 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/NBA(Salary).xlsx')
    merged_df = pd.merge(df1, df2, on='Player', how='inner')
    merged_df['AST - TOV'] = merged_df['AST'] - merged_df['TOV']
    merged_df['BLK + STL'] = merged_df['BLK'] + merged_df['STL']
    features = ['PTS', 'eFG%', 'TRB', 'AST - TOV', 'BLK + STL', '3P%']
    X = merged_df[features]

    imputer = SimpleImputer(strategy="mean")
    X[:] = imputer.fit_transform(X)
    merged_df[features] = X

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
    return merged_df, rf_model, y_test, y_pred

merged_df, rf_model, y_test, y_pred = load_data()

st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose Analysis Mode", ["Random Forest","K-Means"])
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
        "Classify Quality Player",
        "Model Accuracy",
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

if mode == "Random Forest":
    if option == "Quality Players":
        st.header("Quality Player Analysis")
        quality_players = filtered_df[filtered_df['Quality Player'] == 1]
        st.write(f"Number of Quality Players: {len(quality_players)}")
        st.dataframe(quality_players[['Player', 'PTS', 'TRB', 'AST', 'BLK', 'STL', 'Quality Player', 'Salary Player']])

    elif option == "PTS vs Salary":
        st.header("Points (PTS) vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='Salary', y='PTS', hue='Quality Player', palette='viridis', ax=ax)
        sns.regplot(data=filtered_df, x='Salary', y='PTS', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
        ax.set_title("Points (PTS) vs Salary")
        ax.grid(True)
        for i, row in filtered_df.iterrows():
            ax.annotate(i, (row['Salary'], row['PTS']), fontsize=8, alpha=0.7)
        st.pyplot(fig)

    elif option == "eFG% vs Salary":
        st.header("eFG% vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='Salary', y='eFG%', hue='Quality Player', palette='viridis', ax=ax)
        sns.regplot(data=filtered_df, x='Salary', y='eFG%', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
        ax.set_title("eFG% vs Salary")
        ax.grid(True)
        for i, row in filtered_df.iterrows():
            ax.annotate(i, (row['Salary'], row['eFG%']), fontsize=8, alpha=0.7)
        st.pyplot(fig)

    elif option == "TRB vs Salary":
        st.header("Total Rebounds (TRB) vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='Salary', y='TRB', hue='Quality Player', palette='viridis', ax=ax)
        sns.regplot(data=filtered_df, x='Salary', y='TRB', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
        ax.set_title("Total Rebounds (TRB) vs Salary")
        ax.grid(True)
        for i, row in filtered_df.iterrows():
            ax.annotate(i, (row['Salary'], row['TRB']), fontsize=8, alpha=0.7)
        st.pyplot(fig)

    elif option == "3P% vs Salary":
        st.header("3P% vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='Salary', y='3P%', hue='Quality Player', palette='viridis', ax=ax)
        sns.regplot(data=filtered_df, x='Salary', y='3P%', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
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
        sns.regplot(data=filtered_df, x='Salary', y='AST - TOV', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
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
        sns.regplot(data=filtered_df, x='Salary', y='BLK + STL', scatter=False, ax=ax, color='blue', line_kws={"label": "Linear Regression","linestyle": "dashed"})
        ax.set_title("BLK + STL vs Salary")
        ax.grid(True)
        for i, row in filtered_df.iterrows():
            ax.annotate(i, (row['Salary'], row['BLK + STL']), fontsize=8, alpha=0.7)
        st.pyplot(fig)
        st.pyplot(fig)

    if option == "Classify Quality Player":
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

    elif option == "Model Accuracy":
        df1 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/Data_NBA(2).xlsx')
        df2 = pd.read_excel('https://raw.githubusercontent.com/KyanDV/NBA_Analysis/main/NBA(Salary).xlsx')
        merged_df = pd.merge(df1, df2, on='Player', how='inner')
        merged_df['AST - TOV'] = merged_df['AST'] - merged_df['TOV']
        merged_df['BLK + STL'] = merged_df['BLK'] + merged_df['STL']
        features = ['PTS', 'eFG%', 'TRB', 'AST - TOV', 'BLK + STL', '3P%']
        X = merged_df[features]

        imputer = SimpleImputer(strategy="mean")
        X[:] = imputer.fit_transform(X)
        merged_df[features] = X

        # Threshold
        thresholds = {stat: X[stat].mean() for stat in features}
        merged_df['Quality Player'] = (
            (merged_df['PTS'] > thresholds['PTS']) &
            (merged_df['TRB'] > thresholds['TRB']) &
            (merged_df['AST - TOV'] > thresholds['AST - TOV']) &
            (merged_df['BLK + STL'] > thresholds['BLK + STL']) &
            (merged_df['eFG%'] > thresholds['eFG%'])
        ).astype(int)
        y = merged_df['Quality Player']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.header("Accuracy Model")
        st.write(f"Random Forest Model Accuracy: {accuracy:.2f}")

elif mode == "K-Means":
    k = st.sidebar.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=3)
    
    if option == "PTS vs Salary":
        X = merged_df[['Salary', 'PTS']]
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("Points (PTS) vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['PTS'], label=f'Cluster {cluster}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('Points')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['PTS']), fontsize=8, alpha=0.7)

        # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['PTS'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        pts_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, pts_pred, color='blue', linestyle='dashed', label='Linear Regression')

        st.pyplot(fig)

    elif option == "eFG% vs Salary":
        X = merged_df[['Salary', 'eFG%']]
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("eFG% vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['eFG%'], label=f'Cluster {cluster}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('eFG%')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['eFG%']), fontsize=8, alpha=0.7)

        # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['eFG%'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        efg_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, efg_pred, color='blue', linestyle='dashed', label='Linear Regression')
        

        st.pyplot(fig)

    elif option == "TRB vs Salary":
        X = merged_df[['Salary', 'TRB']] 
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("Total Rebounds (TRB) vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['TRB'], label=f'Cluster {cluster}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('Total Rebounds (TRB)')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['TRB']), fontsize=8, alpha=0.7)

        # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['TRB'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        pts_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, pts_pred, color='blue', linestyle='dashed', label='Linear Regression')

        st.pyplot(fig)

    elif option == "3P% vs Salary":
        X = merged_df[['Salary', '3P%']] 
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("3P% vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['3P%'], label=f'Cluster {cluster}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('3P%')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['3P%']), fontsize=8, alpha=0.7)

        # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['3P%'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        efg_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, efg_pred, color='blue', linestyle='dashed', label='Linear Regression')

        st.pyplot(fig)

    elif option == "AST - TOV vs Salary":
        X = merged_df[['Salary', 'AST - TOV']] 
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("AST - TOV vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['AST - TOV'], label=f'Cluster {cluster}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('AST - TOV')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['AST - TOV']), fontsize=8, alpha=0.7)

        # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['AST - TOV'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        pts_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, pts_pred, color='blue', linestyle='dashed', label='Linear Regression')

        st.pyplot(fig)

    elif option == "BLK + STL vs Salary":
        X = merged_df[['Salary', 'BLK + STL']]
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.header("BLK + STL vs. Salary")
        fig, ax = plt.subplots(figsize=(10, 6))

        for cluster in range(k):
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            ax.scatter(cluster_data['Salary'], cluster_data['BLK + STL'], label=f'Cluster {cluster}')
            
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='*', label='Centroids')

        ax.set_xlabel('Salary')
        ax.set_ylabel('BLK + STL')
        ax.set_title('KMeans Clustering NBA Players')
        ax.legend()
        ax.grid(True)

        for i, row in merged_df.iterrows():
            ax.annotate(i, (row['Salary'], row['BLK + STL']), fontsize=8, alpha=0.7)

            # Linear Regression
        reg = LinearRegression()
        reg.fit(X[['Salary']], X['BLK + STL'])
        salary_range = np.linspace(X['Salary'].min(), X['Salary'].max(), 100)
        pts_pred = reg.predict(salary_range.reshape(-1, 1))
        ax.plot(salary_range, pts_pred, color='blue', linestyle='dashed', label='Linear Regression')

        st.pyplot(fig)

    elif option == "Quality Players":
        st.header("Quality Player Analysis")
        quality_players = filtered_df[filtered_df['Quality Player'] == 1]
        st.write(f"Number of Quality Players: {len(quality_players)}")
        st.dataframe(quality_players[['Player', 'PTS', 'TRB', 'AST', 'BLK', 'STL', 'Quality Player', 'Salary Player']])
    
    elif option == "Model Accuracy":
        X = merged_df[['Salary', 'PTS']]  
        kmeans = KMeans(n_clusters=k, random_state=0)
        merged_df['Cluster'] = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        silhouette_avg = silhouette_score(X, merged_df['Cluster'])
        inertia = kmeans.inertia_

        st.subheader("K-Means Model Evaluation")
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        st.write(f"Inertia: {inertia:.2f}")

