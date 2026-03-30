import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data/Zomato Restaurant names and Metadata.csv")

print("Columns:", df.columns)

# -------------------------------
# 2. CLEAN DATA
# -------------------------------
df = df.drop_duplicates()

# Clean Cost column
df['Cost'] = df['Cost'].astype(str).str.replace(',', '')
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# Drop nulls
df = df.dropna(subset=['Cost'])

print("Cleaned Shape:", df.shape)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
df['Cost_log'] = np.log1p(df['Cost'])
df['Name_length'] = df['Name'].apply(len)

# -------------------------------
# 4. EDA (15 CHARTS TOTAL)
# -------------------------------

# 1️⃣ Histogram (ONLY ONE)
plt.figure(figsize=(8,5))
sns.histplot(df['Cost'], bins=30)
plt.title("Cost Distribution")
plt.show()

# 2️⃣ Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Cost'])
plt.title("Cost Boxplot")
plt.show()

# 3️⃣ Countplot (Top Cuisines)
plt.figure(figsize=(10,6))
sns.countplot(y=df['Cuisines'], order=df['Cuisines'].value_counts().head(10).index)
plt.title("Top 10 Cuisines")
plt.show()

# 4️⃣ Scatter (Cost vs Name Length)
plt.figure(figsize=(8,5))
sns.scatterplot(x='Cost', y='Name_length', data=df)
plt.title("Cost vs Name Length")
plt.show()

# 5️⃣ Violin Plot
plt.figure(figsize=(8,5))
sns.violinplot(x=df['Cost'])
plt.title("Cost Distribution (Violin)")
plt.show()

# 6️⃣ KDE Plot
plt.figure(figsize=(8,5))
sns.kdeplot(df['Cost'], fill=True)
plt.title("Cost Density")
plt.show()

# 7️⃣ Pie Chart
plt.figure(figsize=(6,6))
df['Cuisines'].value_counts().head(5).plot.pie(autopct='%1.1f%%')
plt.title("Top 5 Cuisines")
plt.ylabel('')
plt.show()

# 8️⃣ Strip Plot
plt.figure(figsize=(8,5))
sns.stripplot(x=df['Cost'])
plt.title("Strip Plot")
plt.show()

# 9️⃣ Swarm Plot
plt.figure(figsize=(8,5))
sns.swarmplot(x=df['Cost'])
plt.title("Swarm Plot")
plt.show()

# -------------------------------
# 5. FEATURE SELECTION
# -------------------------------
X = df[['Cost', 'Cost_log', 'Name_length']]

# -------------------------------
# 6. SCALING
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 7. ELBOW METHOD
# -------------------------------
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 10), wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# -------------------------------
# 8. MODEL BUILDING
# -------------------------------

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)
k_score = silhouette_score(X_scaled, k_labels)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=3)
a_labels = agg.fit_predict(X_scaled)
a_score = silhouette_score(X_scaled, a_labels)

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
d_labels = db.fit_predict(X_scaled)

if len(set(d_labels)) > 1:
    d_score = silhouette_score(X_scaled, d_labels)
else:
    d_score = -1

# -------------------------------
# 9. MODEL COMPARISON
# -------------------------------
print("\nModel Scores:")
print("KMeans:", k_score)
print("Agglomerative:", a_score)
print("DBSCAN:", d_score)

scores = {
    "KMeans": k_score,
    "Agglomerative": a_score,
    "DBSCAN": d_score
}

best_model = max(scores, key=scores.get)
print("\nBest Model:", best_model)

# Assign clusters
if best_model == "KMeans":
    df['cluster'] = k_labels
elif best_model == "Agglomerative":
    df['cluster'] = a_labels
else:
    df['cluster'] = d_labels

# -------------------------------
# 🔥 ADDITIONAL VISUALS (FINAL SET)
# -------------------------------

# 10️⃣ Heatmap (Correlation)
plt.figure(figsize=(8,5))
sns.heatmap(df[['Cost', 'Cost_log', 'Name_length']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 11️⃣ Heatmap (Cuisine vs Cost)
pivot = pd.pivot_table(df, values='Cost', index='Cuisines', aggfunc='mean').head(10)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap='viridis')
plt.title("Average Cost per Cuisine")
plt.show()

# 12️⃣ Heatmap (Cluster Analysis)
cluster_summary = df.groupby('cluster')[['Cost', 'Cost_log', 'Name_length']].mean()

plt.figure(figsize=(8,5))
sns.heatmap(cluster_summary, annot=True, cmap='coolwarm')
plt.title("Cluster-wise Features")
plt.show()

# 13️⃣ Cluster Count
plt.figure(figsize=(6,4))
sns.countplot(x='cluster', data=df)
plt.title("Cluster Distribution")
plt.show()

# 14️⃣ Cluster Scatter
plt.figure(figsize=(8,5))
sns.scatterplot(x='Cost', y='Name_length', hue='cluster', data=df)
plt.title("Cluster Visualization")
plt.show()

# 15️⃣ Pair Plot (IMPORTANT)
sns.pairplot(df[['Cost', 'Cost_log', 'Name_length', 'cluster']], hue='cluster')
plt.show()

# -------------------------------
# 10. FINAL OUTPUT
# -------------------------------
print("\nFinal Data Sample:")
print(df.head())