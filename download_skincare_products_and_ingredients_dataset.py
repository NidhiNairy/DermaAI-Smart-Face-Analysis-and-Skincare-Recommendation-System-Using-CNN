import kagglehub

# Download latest version
path = kagglehub.dataset_download("eward96/skincare-products-and-their-ingredients")

print("Path to dataset files:", path)


import pandas as pd
import os

# Assuming there’s a CSV file inside
csv_file = os.path.join(path, "skincare_products.csv")  # adjust name if different
df = pd.read_csv(csv_file)

print("Number of items:", len(df))
print("Columns:", df.columns)
