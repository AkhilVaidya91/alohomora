import numpy as np
import pandas as pd

filename = 'ecommerce_sellers_with_best_quality.csv'
df = pd.read_csv(filename)

target_column = 'best_quality'

noise_level = 0.1

df[target_column] = df[target_column] + noise_level * np.random.randn(len(df))

df[target_column] = np.clip(df[target_column], 0, 5)
df.head(10)

new_filename = filename.replace('.csv', '_with_noise.csv')
df.to_csv(new_filename, index=False)