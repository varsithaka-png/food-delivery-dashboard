Smart Food Delivery Analytics and Prediction System
Overview

This project is an interactive dashboard built using Streamlit that analyzes food delivery data and predicts whether a delivery will be delayed. It combines data visualization with machine learning to provide insights into delivery performance.

Features
Interactive dashboard with restaurant-level filtering
Key performance indicators (Total Orders, Average Delivery Time, Average Rating)
Visualizations for order patterns and delivery performance
Machine learning model to predict delivery delay
Feature importance analysis for model interpretability
Real-time prediction based on user inputs
Clean and modern UI using custom styling
Machine Learning

The project uses a Random Forest Classifier to predict delivery delay.

Target variable:

Delay (1 = Late, 0 = On Time)

Evaluation metrics:

Accuracy score
Confusion matrix
Dataset

The dataset includes both operational and contextual features:

OrderTime: Hour when the order was placed
DeliveryTime: Time taken for delivery (minutes)
Distance: Distance between restaurant and customer
TrafficLevel: Low, Medium, High
Weather: Clear, Cloudy, Rainy
RiderExperience: Experience level of delivery partner
Rating: Customer rating
Tech Stack
Python
Streamlit
Pandas
Matplotlib
Scikit-learn
