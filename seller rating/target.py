import numpy as np
import pandas as pd

df = pd.read_csv('ecommerce_sellers.csv')

weights = {
    'on_time_delivery_rate': 0.05,
    'flexibility_in_urgent_orders': 0.05,
    'competitive_pricing': 0.4,
    'payment_terms': 0.3,
    'financial_stability': 0.15,
    'quality_consistency': 0.1,
    'percentage_of_returns': -0.3,
    'adherence_to_specifications': 0.1
}

def calculate_quick_delivery(row):
    score = (
        weights['on_time_delivery_rate'] * row['on_time_delivery_rate'] / 20 +
        weights['flexibility_in_urgent_orders'] * row['flexibility_in_urgent_orders'] +
        weights['competitive_pricing'] * (row['competitive_pricing'] + 1) +
        weights['payment_terms'] * row['payment_terms'] +
        weights['financial_stability'] * row['financial_stability'] +
        weights['quality_consistency'] * row['quality_consistency'] +
        weights['percentage_of_returns'] * row['percentage_of_returns'] / 20 +
        weights['adherence_to_specifications'] * row['adherence_to_specifications']
    )
    
    normalized_score = (score - 0) / (5 - 0) * 5
    
    # Clip
    return np.clip(normalized_score, 0, 5)

df['lowest_cost'] = df.apply(calculate_quick_delivery, axis=1)

# print(df[['category', 'lowest_cost']].head(10))

df.to_csv('ecommerce_sellers_with_lowest_cost.csv', index=False)