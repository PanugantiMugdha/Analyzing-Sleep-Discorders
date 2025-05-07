# Predicting Sleep Disorders

## 📌 Problem Statement
Sleep disorders like chronic insomnia and obstructive sleep apnea affect millions globally but are often underdiagnosed due to limited access to clinical resources. This project aims to provide an intelligent, accessible solution that predicts the risk of sleep disorders using health and lifestyle data.

## 🎯 Objective
- Predict sleep disorders using a machine learning model.
- Offer personalized recommendations and early warnings.
- Provide a multilingual chatbot interface for broader reach and usability.

## 📖 Introduction
The fusion of AI and healthcare allows for innovative approaches in early diagnosis. This system leverages a multimodal pipeline—structured data and natural language inputs—to identify individuals at risk and guide them toward better sleep hygiene and medical support.

## 🧠 Algorithm Used: Random Forest
We use the **Random Forest** algorithm due to its:
- High accuracy and robustness against overfitting.
- Ability to handle both numerical and categorical data.
- Interpretability through feature importance ranking.

### Workflow:
1. **Data Preprocessing**: Normalize and encode attributes like age, BMI, stress, heart rate, and blood pressure.
2. **Model Training**: Apply Random Forest on the Sleep Health and Lifestyle Dataset.
3. **Evaluation**: Use accuracy, precision, recall, and F1-score to validate the model.
4. **Chatbot Integration**: Integrate with a GPT-based chatbot for user-friendly interaction and recommendations.

## 🧪 Dataset
- **Source**: Sleep Health and Lifestyle Dataset (Kaggle)
- **Attributes Used**: Age, Gender, Occupation, Sleep Duration, Physical Activity, Stress Levels, BMI, Blood Pressure, Heart Rate

## 🗣️ Features
- Sleep disorder prediction (Insomnia, Apnea, None)
- Personalized feedback based on health parameters
- Multilingual chatbot interface for engagement

## ✅ Conclusion
This project presents a hybrid AI model combining Random Forest and language models to provide an accessible, preventive healthcare tool for sleep disorders. It shows promise for further deployment in wellness applications and remote health monitoring platforms.

## 📂 Folder Structure
