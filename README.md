# Machine Learning Models for Seller Rating System and Product Demand Forecasting

## Project Overview

This project involves the development of two machine learning models:

1. **Seller Rating System**: This model assigns a rating between 0 and 5 to each seller based on their historical performance. The ratings are calculated for three target parameters: delivery time, product quality, and cost.

2. **Product Demand Forecasting**: This model predicts the demand for products in the upcoming four weeks using a SARIMAX model (ARIMA also included for lighter model variation), trained on three years of seasonal per-week sales data.

## Seller Rating System

### Objective

To predict ratings for sellers on three key parameters: delivery time, product quality, and cost, based on multiple performance-related features.

### Input Features

- `on_time_delivery_rate`: Rate at which a seller delivers on time.
- `flexibility_in_urgent_orders`: Seller's ability to accommodate urgent orders.
- `competitive_pricing`: Pricing competitiveness of the seller.
- `payment_terms`: Favorability of the seller's payment terms.
- `financial_stability`: Financial stability of the seller.
- `quality_consistency`: Consistency in product quality delivered by the seller.
- `percentage_of_returns`: Percentage of products returned by customers.
- `adherence_to_specifications`: How well the seller adheres to product specifications.
- `category`: Category of the products sold by the seller.

### Target Parameters

- **Delivery Time Rating**
- **Product Quality Rating**
- **Cost Rating**

### Approach

- The model is trained on a synthetically generated dataset representing the historical performance of sellers.
- Scikit-learn is used for model development and evaluation.

## Product Demand Forecasting

### Objective

To predict the demand for products in the upcoming four weeks, aiding in effective inventory management.

### Data

- The model is trained on three years of seasonal per-week sales data for each product.

### Approach

- A SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) model is implemented for each product.
- Statsmodels library is used for model implementation.

## Libraries Used

- **scikit-learn**: For building and evaluating the seller rating model.
- **statsmodels**: For implementing the SARIMAX model for demand forecasting.

## Conclusion

This project provides a comprehensive solution for evaluating seller performance and forecasting product demand, which can be leveraged in an inventory management system to optimize decision-making.