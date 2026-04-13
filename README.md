Smart Food Delivery Analytics and Prediction System
---------------------------------------------------------------------------
Overview
This project is an interactive dashboard built using Streamlit.
It analyzes food delivery data and predicts whether a delivery will be On Time or Late using machine learning.
---------------------------------------------------------------------------
Features
Interactive dashboard with filters
Key metrics (Total Orders, Average Delivery Time, Rating)
Data visualizations for better insights
Machine Learning model (Random Forest)
Feature importance analysis
Real-time prediction system
Clean and modern user interface
Machine Learning
Model used: Random Forest Classifier
Target variable: Delay
0 = On Time
1 = Late
---------------------------------------------------------------------------
Evaluation methods:
Accuracy Score
Confusion Matrix
---------------------------------------------------------------------------
Dataset
The dataset includes:
OrderTime – Time when order is placed (0–23)
DeliveryTime – Delivery duration in minutes
Distance – Distance between restaurant and customer
TrafficLevel – Low, Medium, High
Weather – Clear, Cloudy, Rainy
RiderExperience – Experience level (1–5)
Rating – Customer rating
Tech Stack
Python
Streamlit
Pandas
Matplotlib
Scikit-learn
---------------------------------------------------------------------------
Project Structure
food-delivery-dashboard/
│
├── app.py
├── data.csv
├── requirements.txt
└── README.md
---------------------------------------------------------------------------
Installation
Clone the repository:
git clone https://github.com/your-username/food-delivery-dashboard.git
cd food-delivery-dashboard

Install dependencies:

pip install -r requirements.txt

Run the project:
streamlit run app.py

---------------------------------------------------------------------------
Usage
Select a restaurant from the sidebar
View metrics and charts
Check model performance
Enter input values for prediction
Get delivery status result
---------------------------------------------------------------------------
Key Insights
Evening hours have more orders
High traffic increases delays
Longer distance leads to slower delivery
Weather affects delivery time
Experienced riders deliver faster
---------------------------------------------------------------------------
Future Improvements
Use real-world dataset
Compare multiple ML models
Deploy online
Improve UI further
---------------------------------------------------------------------------
Author
Varsitha
---------------------------------------------------------------------------
Conclusion
This project shows how data analysis and machine learning can be combined to solve real-world problems like delivery delay prediction.
