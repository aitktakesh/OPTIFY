import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate Synthetic Customer Data
num_customers = 5000

np.random.seed(42)  # For reproducibility
customer_ids = np.arange(1000, 1000 + num_customers)
products_purchased = np.random.randint(1, 15, size=num_customers)  # Between 1 and 15 products
complains = np.random.choice([0, 1], size=num_customers, p=[0.98, 0.02])  # 2% complain
money_spent = products_purchased * np.random.uniform(10, 300, size=num_customers)  # Random spending

# Create DataFrame
customer_data = pd.DataFrame({
    "customer_id": customer_ids,
    "products_purchased": products_purchased,
    "complains": complains,
    "money_spent": money_spent
})

# Step 2: Prepare Data for Clustering
features = customer_data[["products_purchased", "complains", "money_spent"]]

# Standardizing the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Apply K-Means Clustering
num_clusters = 4  # Define number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
customer_data["cluster"] = kmeans.fit_predict(features_scaled)

# Step 4: Save the Clustered Dataset
customer_data.to_csv("customer_clusters.csv", index=False)
print("Clustered dataset saved as 'customer_clusters.csv'")

# Step 5: Visualization of Clusters
plt.figure(figsize=(10, 6))

# Scatter plot for Money Spent vs Products Purchased
scatter = plt.scatter(customer_data["products_purchased"], customer_data["money_spent"], 
                      c=customer_data["cluster"], cmap="viridis", alpha=0.6, edgecolors="k")

plt.xlabel("Products Purchased")
plt.ylabel("Money Spent")
plt.title("Customer Segmentation using K-Means")
plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(num_clusters)])
plt.show()

# Step 6: Display Cluster Distribution
print("Cluster distribution:\n", customer_data["cluster"].value_counts())
