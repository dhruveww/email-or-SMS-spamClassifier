# Spam Classification System

This project implements a spam classification system using **Machine Learning** and **Natural Language Processing (NLP)** techniques. The system classifies text messages as spam or ham (non-spam) based on multiple models, with performance evaluation across various metrics like accuracy and precision.

## Project Overview

The goal of this project is to build a reliable spam detection system that can identify spam messages using several machine learning algorithms and NLP techniques. The system uses **TF-IDF vectorization** for feature extraction and compares different models, including **Naive Bayes**, **Logistic Regression**, **Random Forest**, and **XGBoost**.

The project also includes a **Streamlit**-based web application for real-time spam classification, which has been deployed on **Render.io** for scalable and accessible predictions.

## Key Features

- **Data Preprocessing**: Includes data cleaning, feature engineering, and handling label distributions.
- **Model Training**: Trained and evaluated several machine learning models, including **Multinomial Naive Bayes**, **Logistic Regression**, **Random Forest**, **XGBoost**, and ensemble models.
- **Performance Optimization**: Optimized feature extraction using **TF-IDF** with top 3000 and 5000 features.
- **Hyperparameter Tuning**: Fine-tuned models for improved accuracy and precision.
- **Interactive Web Application**: A **Streamlit** app for real-time spam predictions.
- **Deployment**: The app is deployed on **Render.io** for cloud hosting and scalable access.

## Technologies Used

- **Python**: For data analysis, model training, and web app development.
- **Libraries**: 
  - **Pandas** for data manipulation.
  - **Scikit-learn** for machine learning models and evaluation.
  - **Streamlit** for creating the web application.
  - **TF-IDF** from **Scikit-learn** for feature extraction.
- **Deployment**: 
  - **Render.io** for hosting the Streamlit app.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-classification.git
   cd spam-classification
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app locally
   ```bash
   streamlit run app.py
4. Launch the Streamlit app in your browser:
   - The app will open automatically at `http://localhost:8501` after running the above command.

## Data Preprocessing

Data preprocessing is a crucial step in ensuring the quality and effectiveness of machine learning models. The following steps were taken in this project:

- **Data Cleaning**: Removal of unnecessary columns and standardization of column names for clarity. The data was checked for missing or null values and handled accordingly.
- **Feature Engineering**: The text data was vectorized using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method, converting text into numerical features for machine learning models.
- **Label Distribution Analysis**: We analyzed the distribution of labels to ensure a balanced dataset, which is critical for model performance.

## Model Training & Evaluation

Multiple machine learning models were employed to classify messages as spam or ham:

- **Multinomial Naive Bayes (NB)**: The best-performing model, achieving **96.8% accuracy** and **100% precision**. It performed well in classifying text-based data.
- **Logistic Regression (LR)**: A simple and interpretable model used as a baseline for comparison.
- **Random Forest (RF)**: A robust model that combines the predictions of multiple decision trees.
- **XGBoost (XGB)**: A gradient boosting technique that provides high accuracy and efficiency.
- **Ensemble Models**: A combination of base models to improve predictive accuracy.

Hyperparameter tuning was performed to optimize the models using different feature sets (top **3000** and **5000** features). These optimizations helped improve model performance, especially in terms of precision and accuracy.

### Model Performance Comparison

| Model                        | Accuracy (CV) | Precision (CV) | Accuracy (TF-IDF) | Precision (TF-IDF) | Accuracy (TF3000) | Precision (TF3000) | Accuracy (TF5000) | Precision (TF5000) |
|------------------------------|---------------|----------------|-------------------|--------------------|-------------------|--------------------|-------------------|--------------------|
| **Multinomial Naive Bayes (NB)**  | 0.970019      | 0.880000       | 0.948743          | 1.000000           | 0.968085          | 1.000000           | 0.960348          | 1.000000           |
| **Logistic Regression (LR)**      | 0.969052      | 0.959350       | 0.942940          | 0.870690           | 0.945841          | 0.867769           | 0.945841          | 0.873950           |
| **Extra Tree Classifier (ETC)**   | 0.969052      | 0.991304       | 0.970986          | 0.983193           | 0.974855          | 0.983740           | 0.971954          | 0.975410           |
| **Bernoulli Naive Bayes (BNB)**   | 0.968085      | 0.974576       | 0.968085          | 0.974576           | 0.972921          | 0.991597           | 0.972921          | 0.983471           |
| **XGBoost (xgb)**                | 0.965184      | 0.965812       | 0.970986          | 0.960000           | 0.965184          | 0.950413           | 0.968085          | 0.959016           |
| **Random Forest (RF)**           | 0.963250      | 0.990826       | 0.970986          | 0.991453           | 0.974855          | 0.991736           | 0.970019          | 0.991379           |
| **AdaBoost (AdaBoost)**          | 0.956480      | 0.890625       | 0.955513          | 0.938053           | 0.957447          | 0.910569           | 0.960348          | 0.933333           |
| **Gradient Boosting (GBDT)**     | 0.941006      | 0.903846       | 0.949710          | 0.951456           | 0.954545          | 0.971154           | 0.950677          | 0.951923           |
| **Decision Tree (DT)**           | 0.912959      | 0.816092       | 0.935203          | 0.867925           | 0.933269          | 0.865385           | 0.936170          | 0.849558           |
| **K-Nearest Neighbors (KN)**     | 0.899420      | 1.000000       | 0.892650          | 1.000000           | 0.904255          | 0.979167           | 0.902321          | 0.978261           |
| **Support Vector Machine (SVC)** | 0.849130      | 0.462069       | 0.962282          | 0.914062           | 0.966151          | 0.898551           | 0.961315          | 0.906977           |
| **Gaussian Naive Bayes (GNB)**   | 0.808511      | 0.414239       | 0.804642          | 0.406557           | 0.804642          | 0.406557           | 0.804642          | 0.406557           |


## Web Application

An interactive web application was developed using **Streamlit** to provide real-time spam classification. This app allows users to enter a text message and immediately receive a prediction (spam or ham). The app was deployed on **Render.io**, ensuring scalability and accessibility.

### How to Use the Web Application

1. **Run Streamlit**:
   - After installing the dependencies, use the command `streamlit run app.py` to start the app locally.
   
2. **Input Message**:
   - Once the app is running, go to your browser (`http://localhost:8501`) and enter a message to check if it's spam or ham.

3. **Cloud Deployment**:
   - The app is hosted on **Render.io**, making it accessible for cloud-based real-time predictions. Access it [here](https://your-render-deployment-url).

## Deployment

The project is hosted on **Render.io** to make the spam classification model scalable and accessible. Render provides a cloud environment that supports the deployment of Python applications, ensuring real-time predictions from anywhere with minimal infrastructure management.

### Deployment Steps:

1. Set up a **Render.io** account and follow their deployment instructions for **Streamlit apps**.
2. Push the repository to **GitHub**.
3. Link the GitHub repository to **Render.io** and deploy the app to the cloud.

## Conclusion

This spam classification system utilizes advanced **NLP** and **machine learning** techniques to classify messages accurately. With **TF-IDF** vectorization and the use of multiple models such as **Naive Bayes**, **Random Forest**, and **XGBoost**, the system achieves high accuracy and precision. The **Streamlit** web application provides a user-friendly interface for real-time predictions, and the deployment on **Render.io** ensures scalability and easy access for end-users.

## Future Improvements

- **Advanced NLP Models**: Experimenting with models like **BERT** or **GPT-3** for better accuracy in spam detection.
- **Batch Prediction**: Enabling the ability to classify multiple messages at once in the web app.
- **Model Explainability**: Implementing tools to interpret model predictions, helping users understand why a message is classified as spam or ham.
- **User Interface Enhancements**: Improving the visual elements of the Streamlit app for a more polished user experience.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

