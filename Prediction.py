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

# Function to collect user symptoms - ENHANCED VERSION
def collect_symptoms():
    try:
        print("\n--- Enter Your Sleep Symptoms ---")

        # Create user-friendly prompts with examples and explanations
        print("\nSLEEP PATTERN SYMPTOMS:")
        sleep_duration = float(input("How many hours do you sleep per night on average? (e.g. 6.5): "))
        quality_of_sleep = int(input("Rate your sleep quality from 1-10 (1=very poor, 10=excellent): "))

        print("\nPHYSICAL SYMPTOMS:")
        snoring = input("Do you snore? (Yes/No): ").capitalize()
        heart_rate = int(input("What is your average resting heart rate in BPM? (60-100 is typical): "))

        print("\nLIFESTYLE FACTORS:")
        physical_activity = int(input("How many minutes of exercise do you get daily? (e.g. 30): "))
        daily_steps = int(input("Approximately how many steps do you take daily? (e.g. 5000): "))
        stress_level = int(input("Rate your stress level from 1-10 (1=very low, 10=extremely high): "))

        print("\nGENERAL INFORMATION:")
        age = int(input("What is your age? "))
        gender = input("What is your gender? (Male/Female): ").capitalize()

        print("\nADDITIONAL SYMPTOMS (optional):")
        print("Select any additional symptoms you experience:")

        daytime_sleepiness = input("Do you experience excessive daytime sleepiness? (Yes/No): ").capitalize()
        difficulty_falling_asleep = input("Do you have difficulty falling asleep? (Yes/No): ").capitalize()
        night_awakenings = input("Do you frequently wake up during the night? (Yes/No): ").capitalize()
        early_morning_awakening = input("Do you wake up too early and can't fall back asleep? (Yes/No): ").capitalize()
        breathing_pauses = input("Has anyone observed you stop breathing during sleep? (Yes/No): ").capitalize()

        # Validate inputs
        if gender not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        if quality_of_sleep < 1 or quality_of_sleep > 10:
            raise ValueError("Quality of sleep must be between 1 and 10")
        if stress_level < 1 or stress_level > 10:
            raise ValueError("Stress level must be between 1 and 10")
        if snoring not in ['Yes', 'No']:
            raise ValueError("Snoring must be 'Yes' or 'No'")

        # Convert categorical variables for model input
        gender_numeric = 1 if gender == 'Female' else 0
        snoring_numeric = 1 if snoring == 'Yes' else 0

        # Create a DataFrame with user input for the model
        # Note: We're only using the core features the model was trained on
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

        # Store original input with additional symptoms for comprehensive analysis
        original_input = {
            'Age': age,
            'Gender': gender,
            'Sleep_Duration': sleep_duration,
            'Quality_of_Sleep': quality_of_sleep,
            'Physical_Activity': physical_activity,
            'Stress_Level': stress_level,
            'Heart_Rate': heart_rate,
            'Daily_Steps': daily_steps,
            'Snoring': snoring,
            'Daytime_Sleepiness': daytime_sleepiness,
            'Difficulty_Falling_Asleep': difficulty_falling_asleep,
            'Night_Awakenings': night_awakenings,
            'Early_Morning_Awakening': early_morning_awakening,
            'Breathing_Pauses': breathing_pauses
        }

        print("\nThank you for providing your symptoms. Analyzing...")
        return user_data, original_input

    except ValueError as e:
        print(f"Error: {e}")
        return None, None

# Function to predict and explain - ENHANCED with symptom analysis
def predict_disorder(model, user_data, original_input, feature_names):
    # Ensure data has all required features in correct order
    user_data = user_data[feature_names]

    # Get prediction
    prediction = model.predict(user_data)[0]

    # Get probability scores
    proba = model.predict_proba(user_data)[0]
    disorder_names = model.classes_

    # Display results
    print("\n" + "="*50)
    print("SLEEP DISORDER ANALYSIS RESULTS")
    print("="*50)

    print(f"\nBased on your symptoms, you may have: {prediction}")
    print("\nProbability breakdown:")

    # Show probabilities in a formatted way
    for disorder, prob in zip(disorder_names, proba):
        bar_length = int(prob * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        print(f"{disorder:12}: {bar} {int(prob*100)}%")

    # Extract key symptoms from user input
    symptoms_detected = []

    # Sleep pattern symptoms
    if original_input['Sleep_Duration'] < 5.5:
        symptoms_detected.append("Short sleep duration")
    elif original_input['Sleep_Duration'] > 9:
        symptoms_detected.append("Excessive sleep duration")

    if original_input['Quality_of_Sleep'] < 5:
        symptoms_detected.append("Poor sleep quality")

    # Physical symptoms
    if original_input['Snoring'] == 'Yes':
        symptoms_detected.append("Snoring")
    if original_input['Heart_Rate'] > 85:
        symptoms_detected.append("Elevated heart rate")
    if original_input.get('Breathing_Pauses') == 'Yes':
        symptoms_detected.append("Breathing pauses during sleep")

    # Lifestyle factors
    if original_input['Stress_Level'] > 7:
        symptoms_detected.append("High stress levels")
    if original_input['Physical_Activity'] < 30:
        symptoms_detected.append("Low physical activity")

    # Additional symptoms
    if original_input.get('Daytime_Sleepiness') == 'Yes':
        symptoms_detected.append("Excessive daytime sleepiness")
    if original_input.get('Difficulty_Falling_Asleep') == 'Yes':
        symptoms_detected.append("Difficulty falling asleep")
    if original_input.get('Night_Awakenings') == 'Yes':
        symptoms_detected.append("Frequent night awakenings")
    if original_input.get('Early_Morning_Awakening') == 'Yes':
        symptoms_detected.append("Early morning awakening")

    # Provide personalized symptom analysis
    print("\n" + "-"*50)
    print("SYMPTOM ANALYSIS")
    print("-"*50)
    if symptoms_detected:
        print("Key symptoms identified:")
        for symptom in symptoms_detected:
            print(f"• {symptom}")
    else:
        print("No significant risk factors identified in your input.")

    # Disorder-specific analysis
    print("\n" + "-"*50)
    print(f"ABOUT {prediction.upper()}")
    print("-"*50)

    if prediction == "None":
        print("Your symptoms suggest no specific sleep disorder.")
        print("This means your sleep patterns are likely within normal ranges.")
        print("However, you should still maintain good sleep hygiene practices.")

    elif prediction == "Insomnia":
        print("Insomnia is characterized by difficulty falling or staying asleep.")
        print("Common symptoms include:")
        print("• Trouble falling asleep at night")
        print("• Waking up during the night")
        print("• Waking up too early")
        print("• Not feeling well-rested after a night's sleep")
        print("• Daytime tiredness or sleepiness")
        print("• Irritability, depression, or anxiety")
        print("• Difficulty paying attention or focusing")
        print("• Increased errors or accidents")

    elif prediction == "Sleep Apnea":
        print("Sleep apnea is a disorder where breathing repeatedly stops and starts during sleep.")
        print("Common symptoms include:")
        print("• Loud snoring")
        print("• Episodes of stopped breathing during sleep")
        print("• Gasping for air during sleep")
        print("• Waking up with a dry mouth")
        print("• Morning headache")
        print("• Difficulty staying asleep")
        print("• Excessive daytime sleepiness")
        print("• Difficulty paying attention while awake")

    elif prediction == "Hypersomnia":
        print("Hypersomnia involves excessive daytime sleepiness despite good sleep at night.")
        print("Common symptoms include:")
        print("• Extended nighttime sleep")
        print("• Difficulty waking up in the morning")
        print("• Excessive daytime sleepiness")
        print("• Need for daytime naps that don't refresh")
        print("• Brain fog or confusion when waking up")
        print("• Low energy throughout the day")

    # Provide recommendations
    print("\n" + "-"*50)
    print("RECOMMENDATIONS")
    print("-"*50)

    # General recommendations
    print("General sleep improvement tips:")
    print("• Maintain a consistent sleep schedule")
    print("• Create a relaxing bedtime routine")
    print("• Make your bedroom comfortable (cool, dark, quiet)")
    print("• Limit screen time before bed")
    print("• Avoid caffeine and alcohol close to bedtime")

    # Disorder-specific recommendations
    if prediction == "Insomnia":
        print("\nFor insomnia symptoms specifically:")
        print("• Practice relaxation techniques before bed")
        print("• Try cognitive behavioral therapy for insomnia")
        print("• Limit daytime naps")
        print("• Exercise regularly but not close to bedtime")
        print("• Avoid looking at clocks during the night")

    elif prediction == "Sleep Apnea":
        print("\nFor sleep apnea symptoms specifically:")
        print("• Sleep on your side instead of your back")
        print("• Maintain a healthy weight")
        print("• Avoid alcohol and smoking")
        print("• Consider consulting a doctor about CPAP therapy")
        print("• Regular exercise can help improve breathing")

    elif prediction == "Hypersomnia":
        print("\nFor hypersomnia symptoms specifically:")
        print("• Maintain a strict wake-up schedule")
        print("• Take short, scheduled naps when needed")
        print("• Get regular exercise")
        print("• Avoid medications that cause drowsiness")
        print("• Consider light therapy to regulate sleep cycles")

    print("\nIMPORTANT: This analysis is for informational purposes only.")
    print("Please consult with a healthcare professional for proper diagnosis and treatment.")

    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    # Display visual if not in Colab or explicitly set
    if not is_colab or hasattr(plt, 'show'):
        # Visualize the prediction probabilities
        plt.figure(figsize=(10, 5))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = plt.bar(disorder_names, proba, color=colors[:len(disorder_names)])
        plt.title('Sleep Disorder Prediction Probabilities')
        plt.xlabel('Sleep Disorder')
        plt.ylabel('Probability')
        plt.ylim(0, 1)

        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.0%}', ha='center', va='bottom')

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
    print("This system uses machine learning to predict sleep disorders based on your symptoms.")

    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    # Simplified options - focus on symptoms input
    print("\nI'll help you analyze your sleep symptoms and predict potential sleep disorders.")

    # If in Colab, offer to use Kaggle but default to synthetic
    use_kaggle = False
    if is_colab:
        kaggle_choice = input("\nDo you want to use the real dataset from Kaggle? (y/n, default: n): ").lower()
        if kaggle_choice == 'y':
            print("\nYou'll need to upload your kaggle.json API key when prompted.")
            use_kaggle = True

    # Load dataset and train model
    print("\nLoading dataset and training model...")
    data = load_dataset(use_kaggle=use_kaggle)
    model, feature_names = train_model(data)

    # Main loop - focus on symptom input
    while True:
        print("\n" + "="*50)
        print("SLEEP DISORDER SYMPTOM ANALYSIS")
        print("="*50)
        print("Please enter your symptoms and health data for analysis.")

        # Collect user symptoms - this is the main focus
        user_data, original_input = collect_symptoms()

        if user_data is not None:
            # Predict and explain
            predict_disorder(model, user_data, original_input, feature_names)

        # Ask if user wants to continue
        choice = input("\nWould you like to analyze another set of symptoms? (y/n): ")
        if choice.lower() != 'y':
            break

    print("\nThank you for using the Sleep Disorder Prediction System!")
    print("Remember: This is not a substitute for professional medical advice.")

# Run the program
if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
        print("\n" + "="*60)
        print("SLEEP DISORDER PREDICTION SYSTEM - GOOGLE COLAB VERSION")
        print("="*60)
        print("This code is optimized to run directly in Google Colab!")
        print("Dataset source: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
        print("\nNo download required - just follow the prompts to input your symptoms.")
        print("If you want to use the real Kaggle dataset instead of synthetic data,")
        print("you'll need your Kaggle API key (kaggle.json).")
    except:
        is_colab = False
        print("\n" + "="*50)
        print("SLEEP DISORDER PREDICTION SYSTEM")
        print("="*50)
        print("Dataset source: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")

    main()
