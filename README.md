# 🍽️ Zomato Market Segmentation using Unsupervised ML

## 📌 Project Overview

This project focuses on analyzing Zomato restaurant data using **Exploratory Data Analysis (EDA)** and **unsupervised machine learning techniques**. The goal is to identify hidden patterns and segment restaurants based on cost and derived features.

---

## 🎯 Objectives

* Perform detailed **EDA using matplotlib and seaborn**
* Engineer meaningful features from raw data
* Apply **clustering algorithms** to group restaurants
* Compare models and identify the best clustering approach
* Extract actionable insights from the data

---

## 📂 Dataset

The project uses:

* **Zomato Restaurant Metadata Dataset**
* (Optional) Review dataset for additional context

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🔍 Exploratory Data Analysis

We performed extensive EDA including:

* Cost distribution analysis
* Cuisine frequency analysis
* Boxplots, violin plots, KDE plots
* Heatmaps for correlation and feature relationships
* Pairplot for multi-variable relationships

---

## 🤖 Machine Learning Models

This is an **unsupervised learning project**, so we used:

* **K-Means Clustering**
* **Agglomerative Clustering**
* **DBSCAN**

---

## 📊 Model Evaluation

Since there are no labels, we used:

* **Silhouette Score** to evaluate clustering performance
* Visual inspection of cluster separation

---

## 🏆 Results

* K-Means performed best in most cases
* Clear segmentation of restaurants into groups like:

  * Budget-friendly
  * Mid-range
  * Premium

---

## 📈 Key Insights

* Most restaurants fall in lower cost ranges
* Few high-cost outliers exist
* Certain cuisines dominate the market
* Clustering revealed distinct restaurant segments

---

## 📁 Project Structure

```
zomato-market-segmentation-ml/
│
├── data/
│   └── datasets
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 🎤 Conclusion

This project demonstrates how **EDA + clustering** can uncover meaningful patterns in real-world data without relying on labeled outputs.

---


