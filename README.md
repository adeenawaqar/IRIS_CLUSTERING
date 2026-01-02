# 🌸 Iris Flower Clustering App

1) This project uses Unsupervised Machine Learning to group Iris flowers into different clusters based on their physical measurements. 
2) The application is built with Python and deployed via Streamlit.

**📝 Project Introduction**

1) The purpose of this app is to demonstrate how clustering algorithms like K-Means and DBSCAN work on real-world datasets. 
2) By analyzing features such as sepal and petal dimensions, the model identifies patterns and groups flowers without using pre-defined labels.

**Key Features**
1) **Dual Model Comparison:** Provides performance analysis between K-Means and DBSCAN using Silhouette Scores.

2) **Real-time Prediction:** Users can input custom flower measurements to instantly see which cluster the flower belongs to.

3) **Data Scaling:** Built-in StandardScaler to ensure all features are normalized for accurate clustering.

4) **User-Friendly Interface:** A clean and interactive UI built with Streamlit for seamless exploration.


# 📊 Model Results

Based on the training on the Iris dataset, the models achieved the following performance:

   Models        Silhouette Score                  Best For

1) K-Means	       0.4590	                   General grouping and new predictions

2) DBSCAN   	     0.3492	                   Identifying outliers and dense regions


# How to Use

1) Enter the Sepal Length, Sepal Width, Petal Length, and Petal Width.

2) Click the "Identify Cluster" button.

3) The app will return the Cluster Number (1, 0, or 2) and the most likely Iris Species (Setosa, Versicolor, or Virginica).
