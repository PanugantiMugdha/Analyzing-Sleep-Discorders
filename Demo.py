# Google Colab Notebook Setup - Add this at the top of your notebook
# %%capture
# !pip install -q pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Check if running in Google Colab
try:
    import google.colab
    print("Running in Google Colab")
    # Configure Colab-specific settings
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    %matplotlib inline
except:
    print("Not running in Google Colab")

# Function to load and explore the dataset
def load_dataset(use_kaggle=False):
    # Option to load directly from Kaggle in Google Colab
    if use_kaggle:
        try:
            # Install kaggle if not already installed
            !pip install -q kaggle

            # Set up Kaggle API credentials
            # You need to upload your kaggle.json to Colab
            # If not already uploaded, this will create a prompt
            from google.colab import files
            try:
                files.upload()  # Upload kaggle.json if needed
            except:
                pass

            # Make directory for kaggle credentials if it doesn't exist
            !mkdir -p ~/.kaggle
            !cp kaggle.json ~/.kaggle/
            !chmod 600 ~/.kaggle/kaggle.json

            # Download the dataset directly
            !kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset
            !unzip -q sleep-health-and-lifestyle-dataset.zip

            # Read the CSV
            data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
            print("Successfully loaded dataset from Kaggle!")
            return data
        except Exception as e:
            print(f"Error loading from Kaggle: {e}")
            print("Falling back to synthetic data...")

    # For demo purposes without Kaggle API, create synthetic dataset
    print("Creating synthetic dataset based on the Sleep Health and Lifestyle Dataset structure...")
    # Original dataset is available at: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

    # Create synthetic data
    np.random.seed(42)
    n_samples = 400

    # Features
    age = np.random.randint(18, 80, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    sleep_duration = np.random.normal(7, 1.5, n_samples)
    sleep_duration = np.clip(sleep_duration, 2, 12)  # Realistic constraints
    quality_of_sleep = np.random.randint(1, 11, n_samples)
    physical_activity = np.random.randint(10, 100, n_samples)
    stress_level = np.random.randint(1, 11, n_samples)
    heart_rate = np.random.normal(75, 10, n_samples)
    heart_rate = np.clip(heart_rate, 50, 110)  # Realistic constraints
    daily_steps = np.random.normal(7000, 2000, n_samples)
    daily_steps = np.clip(daily_steps, 1000, 15000)  # Realistic constraints
    snoring = np.random.choice(['Yes', 'No'], n_samples)

    # Create some correlation between features and sleep disorders
    sleep_disorder = []
    for i in range(n_samples):
        if sleep_duration[i] < 5.5 and stress_level[i] > 7 and quality_of_sleep[i] < 5:
            disorder = np.random.choice(['Insomnia', 'Sleep Apnea'], p=[0.7, 0.3])
        elif sleep_duration[i] > 9 and physical_activity[i] < 30:
            disorder = np.random.choice(['Hypersomnia', 'Sleep Apnea'], p=[0.6, 0.4])
        elif heart_rate[i] > 85 and quality_of_sleep[i] < 6 and snoring[i] == 'Yes':
            disorder = np.random.choice(['Sleep Apnea', 'Insomnia'], p=[0.8, 0.2])
        elif age[i] > 60 and stress_level[i] > 7:
            disorder = np.random.choice(['Insomnia', 'Sleep Apnea', 'Hypersomnia'], p=[0.5, 0.3, 0.2])
        else:
            disorder = 'None'

        sleep_disorder.append(disorder)

    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Sleep_Duration': sleep_duration,
        'Quality_of_Sleep': quality_of_sleep,
        'Physical_Activity': physical_activity,
        'Stress_Level': stress_level,
        'Heart_Rate': heart_rate,
        'Daily_Steps': daily_steps,
        'Snoring': snoring,
        'Sleep_Disorder': sleep_disorder
    })

    # Add a message about the dataset
    print(f"Dataset created with {n_samples} samples")
    print(f"Sleep disorder distribution: {data['Sleep_Disorder'].value_counts().to_dict()}")

    return data

# Function to explore and visualize dataset
def explore_dataset(data):
    # Check if running in Colab for optimal display
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    print("\n--- Dataset Overview ---")
    print(f"Dataset shape: {data.shape}")
    print("\nSample data:")
    print(data.head())

    print("\nSleep disorder distribution:")
    disorder_counts = data['Sleep_Disorder'].value_counts()
    print(disorder_counts)

    # For Colab, make sure plots display properly
    if is_colab:
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
        %matplotlib inline

    # Create distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sleep_Disorder', data=data)
    plt.title('Sleep Disorder Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sleep duration vs quality of sleep by disorder
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Sleep_Duration', y='Quality_of_Sleep', hue='Sleep_Disorder', data=data, palette='viridis')
    plt.title('Sleep Duration vs Quality of Sleep by Disorder')
    plt.tight_layout()
    plt.show()

    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Function to prepare data and train model
def train_model(data):
    # Convert categorical variables to numerical
    df = data.copy()
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Snoring'] = df['Snoring'].map({'No': 0, 'Yes': 1})

    # Prepare features and target
    X = df.drop('Sleep_Disorder', axis=1)
    y = df['Sleep_Disorder']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"\n--- Model Performance ---")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Sleep Disorder Prediction')
    plt.tight_layout()
    plt.show()

    return model, X.columns

# Function to collect user symptoms
def collect_symptoms():
    try:
        print("\n--- Sleep Disorder Prediction System ---")
        print("Please enter your symptoms and health metrics:")

        age = int(input("Age: "))
        gender = input("Gender (Male/Female): ").capitalize()
        sleep_duration = float(input("Average sleep duration (hours per night): "))
        quality_of_sleep = int(input("Quality of sleep (1-10, 10 being best): "))
        physical_activity = int(input("Physical activity level (minutes per day): "))
        stress_level = int(input("Stress level (1-10, 10 being highest): "))
        heart_rate = int(input("Average resting heart rate (BPM): "))
        daily_steps = int(input("Average daily steps: "))
        snoring = input("Do you snore? (Yes/No): ").capitalize()

        # Validate inputs
        if gender not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        if quality_of_sleep < 1 or quality_of_sleep > 10:
            raise ValueError("Quality of sleep must be between 1 and 10")
        if stress_level < 1 or stress_level > 10:
            raise ValueError("Stress level must be between 1 and 10")
        if snoring not in ['Yes', 'No']:
            raise ValueError("Snoring must be 'Yes' or 'No'")

        # Convert categorical variables
        gender_numeric = 1 if gender == 'Female' else 0
        snoring_numeric = 1 if snoring == 'Yes' else 0

        # Create a DataFrame with user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_numeric],
            'Sleep_Duration': [sleep_duration],
            'Quality_of_Sleep': [quality_of_sleep],
            'Physical_Activity': [physical_activity],
            'Stress_Level': [stress_level],
            'Heart_Rate': [heart_rate],
            'Daily_Steps': [daily_steps],
            'Snoring': [snoring_numeric]
        })

        return user_data, {
            'Age': age,
            'Gender': gender,
            'Sleep_Duration': sleep_duration,
            'Quality_of_Sleep': quality_of_sleep,
            'Physical_Activity': physical_activity,
            'Stress_Level': stress_level,
            'Heart_Rate': heart_rate,
            'Daily_Steps': daily_steps,
            'Snoring': snoring
        }

    except ValueError as e:
        print(f"Error: {e}")
        return None, None

# Function to predict and explain
def predict_disorder(model, user_data, original_input, feature_names):
    # Ensure data has all required features in correct order
    user_data = user_data[feature_names]

    # Get prediction
    prediction = model.predict(user_data)[0]

    # Get probability scores
    proba = model.predict_proba(user_data)[0]
    disorder_names = model.classes_

    # Display results
    print("\n--- Sleep Disorder Prediction Results ---")
    print(f"Predicted sleep disorder: {prediction}")
    print("\nProbability breakdown:")

    for disorder, prob in zip(disorder_names, proba):
        print(f"- {disorder}: {prob:.2f} ({int(prob*100)}%)")

    # Extract key risk factors from user input
    risk_factors = []
    if original_input['Sleep_Duration'] < 5.5:
        risk_factors.append("Short sleep duration")
    elif original_input['Sleep_Duration'] > 9:
        risk_factors.append("Excessive sleep duration")

    if original_input['Quality_of_Sleep'] < 5:
        risk_factors.append("Poor sleep quality")

    if original_input['Stress_Level'] > 7:
        risk_factors.append("High stress levels")

    if original_input['Physical_Activity'] < 30:
        risk_factors.append("Low physical activity")

    if original_input['Heart_Rate'] > 85:
        risk_factors.append("Elevated heart rate")

    if original_input['Snoring'] == 'Yes':
        risk_factors.append("Snoring")

    # Provide personalized analysis
    print("\nAnalysis of your symptoms:")
    if risk_factors:
        print("Key risk factors identified:")
        for factor in risk_factors:
            print(f"- {factor}")
    else:
        print("No significant risk factors identified in your input.")

    # Provide some general advice
    print("\nGeneral advice:")
    if prediction == "None":
        print("Your symptoms suggest no specific sleep disorder, but maintaining good sleep hygiene is always important.")
    elif prediction == "Insomnia":
        print("Your symptoms suggest possible insomnia. Consider:")
        print("- Establishing a regular sleep schedule")
        print("- Reducing screen time before bed")
        print("- Creating a comfortable sleep environment")
        print("- Limiting caffeine and alcohol consumption")
    elif prediction == "Sleep Apnea":
        print("Your symptoms suggest possible sleep apnea. Consider:")
        print("- Consulting with a healthcare provider for proper diagnosis")
        print("- Sleeping on your side instead of your back")
        print("- Maintaining a healthy weight")
        print("- Using a humidifier in your bedroom")
    elif prediction == "Hypersomnia":
        print("Your symptoms suggest possible hypersomnia. Consider:")
        print("- Maintaining a consistent sleep schedule")
        print("- Getting regular exercise (but not right before bed)")
        print("- Avoiding alcohol and sedatives")
        print("- Consulting with a healthcare provider")

    print("\nIMPORTANT: This prediction is not a medical diagnosis. Please consult with a healthcare professional for proper evaluation.")

    # Visualize the prediction probabilities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(disorder_names), y=proba)
    plt.title('Sleep Disorder Prediction Probabilities')
    plt.xlabel('Sleep Disorder')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()

# Demo function with predefined inputs
def run_demo():
    print("\n--- RUNNING DEMO MODE ---")
    print("Loading dataset...")
    data = load_dataset()

    print("\nTraining model...")
    model, feature_names = train_model(data)

    # Demo case 1: Insomnia-like symptoms
    demo_case1 = {
        'Age': 35,
        'Gender': 'Female',
        'Sleep_Duration': 4.5,
        'Quality_of_Sleep': 3,
        'Physical_Activity': 25,
        'Stress_Level': 9,
        'Heart_Rate': 82,
        'Daily_Steps': 4500,
        'Snoring': 'No'
    }

    # Convert to DataFrame with numeric values
    demo_df1 = pd.DataFrame({
        'Age': [demo_case1['Age']],
        'Gender': [1 if demo_case1['Gender'] == 'Female' else 0],
        'Sleep_Duration': [demo_case1['Sleep_Duration']],
        'Quality_of_Sleep': [demo_case1['Quality_of_Sleep']],
        'Physical_Activity': [demo_case1['Physical_Activity']],
        'Stress_Level': [demo_case1['Stress_Level']],
        'Heart_Rate': [demo_case1['Heart_Rate']],
        'Daily_Steps': [demo_case1['Daily_Steps']],
        'Snoring': [1 if demo_case1['Snoring'] == 'Yes' else 0]
    })

    print("\n\n===== DEMO CASE 1: HIGH STRESS, POOR SLEEP QUALITY =====")
    print("Input parameters:")
    for key, value in demo_case1.items():
        print(f"- {key}: {value}")

    predict_disorder(model, demo_df1, demo_case1, feature_names)

    # Demo case 2: Sleep Apnea-like symptoms
    demo_case2 = {
        'Age': 52,
        'Gender': 'Male',
        'Sleep_Duration': 7.2,
        'Quality_of_Sleep': 4,
        'Physical_Activity': 20,
        'Stress_Level': 6,
        'Heart_Rate': 88,
        'Daily_Steps': 3800,
        'Snoring': 'Yes'
    }

    # Convert to DataFrame with numeric values
    demo_df2 = pd.DataFrame({
        'Age': [demo_case2['Age']],
        'Gender': [1 if demo_case2['Gender'] == 'Female' else 0],
        'Sleep_Duration': [demo_case2['Sleep_Duration']],
        'Quality_of_Sleep': [demo_case2['Quality_of_Sleep']],
        'Physical_Activity': [demo_case2['Physical_Activity']],
        'Stress_Level': [demo_case2['Stress_Level']],
        'Heart_Rate': [demo_case2['Heart_Rate']],
        'Daily_Steps': [demo_case2['Daily_Steps']],
        'Snoring': [1 if demo_case2['Snoring'] == 'Yes' else 0]
    })

    print("\n\n===== DEMO CASE 2: SNORING, ELEVATED HEART RATE =====")
    print("Input parameters:")
    for key, value in demo_case2.items():
        print(f"- {key}: {value}")

    predict_disorder(model, demo_df2, demo_case2, feature_names)

    print("\n--- End of Demo ---")
    print("In a real application, you would use the actual Sleep Health and Lifestyle Dataset")
    print("Dataset link: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")

    # Check if running in Colab
    try:
        import google.colab
        print("\nSince you're running this in Google Colab, you can access the dataset directly:")
        print("1. Run the cell with the following code to install and set up kaggle:")
        print("   !pip install kaggle")
        print("   from google.colab import files")
        print("   files.upload()  # Upload your kaggle.json")
        print("   !mkdir -p ~/.kaggle")
        print("   !cp kaggle.json ~/.kaggle/")
        print("   !chmod 600 ~/.kaggle/kaggle.json")
        print("\n2. Then download and use the dataset:")
        print("   !kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset")
        print("   !unzip -q sleep-health-and-lifestyle-dataset.zip")
        print("   data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')")
    except:
        print("Download the dataset and replace the synthetic data generation with:")
        print("data = pd.read_csv('sleep_health_and_lifestyle_dataset.csv')")

# Main function with interactive mode
def main():
    print("=== Sleep Disorder Prediction System ===")
    print("This system uses machine learning to predict sleep disorders based on symptoms and lifestyle factors.")

    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    choice = input("\nDo you want to run in demo mode or interactive mode? (demo/interactive): ").lower()

    # If in Colab, offer to use Kaggle
    use_kaggle = False
    if is_colab and choice != 'demo':
        kaggle_choice = input("\nDo you want to use the real dataset from Kaggle? (y/n): ").lower()
        if kaggle_choice == 'y':
            print("\nYou'll need to upload your kaggle.json API key when prompted.")
            use_kaggle = True

    if choice == 'demo':
        run_demo()
    else:
        # Load dataset and train model
        print("\nLoading dataset and training model...")
        data = load_dataset(use_kaggle=use_kaggle)
        model, feature_names = train_model(data)

        while True:
            # Collect user symptoms
            user_data, original_input = collect_symptoms()

            if user_data is not None:
                # Predict and explain
                predict_disorder(model, user_data, original_input, feature_names)

            # Ask if user wants to continue
            choice = input("\nDo you want to make another prediction? (y/n): ")
            if choice.lower() != 'y':
                break

        print("Thank you for using the Sleep Disorder Prediction System!")

    # Run the program
if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    print("\nSleep Disorder Prediction System")
    print("="*40)
    print("Dataset source: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")

    if is_colab:
        print("\nDetected Google Colab environment!")
        print("You can run this directly in Colab without downloading the dataset.")
        print("To use the real Kaggle dataset, follow these steps:")
        print("1. Get your Kaggle API key from kaggle.com → Account → Create API Token")
        print("2. Upload the kaggle.json file when prompted")
        print("3. Choose 'use_kaggle=True' when asked")
    else:
        print("Download instructions:")
        print("1. Visit the URL above")
        print("2. Click the 'Download' button on Kaggle")
        print("3. Save as 'sleep_health_and_lifestyle_dataset.csv'")
        print("4. Place in the same directory as this script")

    print("="*40)
    main()
