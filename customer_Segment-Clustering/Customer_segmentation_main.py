# Re-import necessary libraries since execution state was reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Regenerate synthetic customer data
num_customers = 5000
np.random.seed(42)

customer_ids = np.arange(1000, 1000 + num_customers)
products_purchased = np.random.randint(1, 15, size=num_customers)
complains = np.random.choice([0, 1], size=num_customers, p=[0.98, 0.02])
money_spent = products_purchased * np.random.uniform(10, 300, size=num_customers)

# Create DataFrame
customer_data = pd.DataFrame({
    "customer_id": customer_ids,
    "products_purchased": products_purchased,
    "complains": complains,
    "money_spent": money_spent
})

# Standardizing the data for clustering
features = customer_data[["products_purchased", "complains", "money_spent"]]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applying K-Means clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
customer_data["cluster"] = kmeans.fit_predict(features_scaled)

# Generate summary statistics
summary_stats = customer_data.describe()

# Display summary statistics
summary_stats
