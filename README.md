### **Online Payment Fraud Detection System**  

A machine learning project to classify online transactions as fraudulent or legitimate.  

#### **Objective**  
- Detect fraudulent transactions to minimize financial losses.  
- Build robust models to balance precision and recall for accurate fraud detection.  
- Deploy the best-performing model for real-time use.  

#### **Features**  
- **Step**: Transaction timestamp.  
- **Type**: Transaction type (e.g., "PAYMENT", "TRANSFER").  
- **Amount**: Transaction value.  
- **Old/New Balance (Sender/Receiver)**: Balances before and after the transaction.  
- **Is Flagged Fraud**: Indicator for flagged transactions.  

#### **Highlights**  
- Preprocessing: One-hot encoding, scaling, and handling class imbalance.  
- Models: Logistic Regression, Random Forest, XGBoost, KNN.  
- Evaluation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  

**Tech Stack**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn.  
