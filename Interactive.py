# Sleep Mystery: 60-Second Interactive Demo
# Perfect for quick audience demonstrations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import time

# Configure display for Colab
display(HTML("<style>.container { width:100% !important; }</style>"))

class SleepMystery:
    def __init__(self):
        """Initialize the Sleep Mystery game"""
        self.model = self.train_quick_model()
        self.current_step = 0
        self.user_data = {}
        self.results = None
        self.sleep_factors = [
            {"name": "sleep_duration", "label": "Hours of Sleep", "min": 2, "max": 12, "default": 7},
            {"name": "sleep_quality", "label": "Sleep Quality", "min": 1, "max": 10, "default": 5},
            {"name": "stress", "label": "Stress Level", "min": 1, "max": 10, "default": 5}
        ]

    def train_quick_model(self):
        """Create and train a simple model for demo purposes"""
        # Simple synthetic dataset
        np.random.seed(42)
        n = 500

        # Features: sleep duration, quality, stress level
        X = np.random.rand(n, 3)
        X[:, 0] = X[:, 0] * 10 + 2  # Sleep duration: 2-12 hours
        X[:, 1] = X[:, 1] * 9 + 1   # Quality: 1-10
        X[:, 2] = X[:, 2] * 9 + 1   # Stress: 1-10

        # Create target with logical rules
        y = np.full(n, "Normal Sleep")

        # Insomnia: low duration, high stress
        insomnia_mask = (X[:, 0] < 5.5) & (X[:, 2] > 7)
        y[insomnia_mask] = "Insomnia"

        # Sleep Apnea: medium duration, low quality
        apnea_mask = (X[:, 0] > 5) & (X[:, 0] < 8) & (X[:, 1] < 4)
        y[apnea_mask] = "Sleep Apnea"

        # Hypersomnia: high duration, medium quality
        hypersomnia_mask = (X[:, 0] > 9) & (X[:, 1] > 3) & (X[:, 1] < 7)
        y[hypersomnia_mask] = "Hypersomnia"

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model

    def start_game(self):
        """Display the intro screen with a start button"""
        clear_output(wait=True)

        # Create title with animation effect
        html_title = """
        <div style="text-align:center; margin:20px; animation: pulse 2s infinite;">
            <h1 style="color:#3d5af1; font-family:Arial; font-size:24px;">✨ SLEEP MYSTERY CHALLENGE ✨</h1>
            <p style="font-size:16px;">Can we guess your sleep condition in 60 seconds?</p>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
        """
        display(HTML(html_title))

        # Create and display start button
        start_button = widgets.Button(
            description='Start the Challenge!',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='50%', height='40px')
        )

        start_button.on_click(lambda b: self.show_step())
        display(widgets.HBox([start_button], layout=widgets.Layout(justify_content='center')))

    def show_step(self):
        """Show the current step in the game"""
        clear_output(wait=True)

        if self.current_step < len(self.sleep_factors):
            factor = self.sleep_factors[self.current_step]
            self.show_question(factor)
        else:
            self.analyze_results()

    def show_question(self, factor):
        """Display a question with a slider"""
        # Create progress indicator
        progress = (self.current_step + 1) / (len(self.sleep_factors) + 1) * 100
        progress_html = f"""
        <div style="width:100%; background-color:#f1f1f1; border-radius:10px; margin:10px 0;">
            <div style="width:{progress}%; background-color:#4CAF50; height:20px; border-radius:10px;
                 text-align:center; line-height:20px; color:white;">
                {int(progress)}%
            </div>
        </div>
        """

        # Question title
        title_html = f"""
        <div style="text-align:center; margin:20px;">
            <h2 style="color:#3d5af1;">Question {self.current_step + 1}/{len(self.sleep_factors)}</h2>
            <p style="font-size:18px;">{factor["label"]}?</p>
        </div>
        """

        # Display progress and title
        display(HTML(progress_html + title_html))

        # Create and display slider
        slider = widgets.IntSlider(
            value=factor["default"],
            min=factor["min"],
            max=factor["max"],
            step=1,
            description='',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='80%')
        )

        # Value descriptions
        if factor["name"] == "sleep_duration":
            low_label = "Few hours"
            high_label = "Many hours"
        elif factor["name"] == "sleep_quality":
            low_label = "Poor quality"
            high_label = "Excellent quality"
        else:  # stress
            low_label = "Very relaxed"
            high_label = "Very stressed"

        # Create labels
        labels = widgets.HBox([
            widgets.HTML(f"<p style='text-align:left'>{low_label}</p>"),
            widgets.HTML(f"<p style='text-align:right'>{high_label}</p>")
        ], layout=widgets.Layout(width='80%', justify_content='space-between'))

        # Create next button
        next_button = widgets.Button(
            description='Next',
            button_style='info',
            icon='arrow-right',
            layout=widgets.Layout(width='40%', height='40px', margin='20px 0 0 0')
        )

        # Store factor name for later use in the button click handler
        factor_name = factor["name"]

        # Create a closure to capture the current values
        def create_click_handler(factor_name, slider_widget):
            def click_handler(b):
                self.save_answer(factor_name, slider_widget.value)
            return click_handler

        # Set up the button's click event with the closure
        next_button.on_click(create_click_handler(factor_name, slider))

        # Display everything
        display(widgets.VBox([
            slider,
            labels,
            widgets.HBox([next_button], layout=widgets.Layout(justify_content='center'))
        ], layout=widgets.Layout(align_items='center')))

    def save_answer(self, name, value):
        """Save the user's answer and move to the next step"""
        self.user_data[name] = value
        self.current_step += 1
        self.show_step()

    def analyze_results(self):
        """Analyze the user's data and show results"""
        # Show loading animation
        loading_html = """
        <div style="text-align:center; margin:30px;">
            <h2 style="color:#3d5af1;">Analyzing your sleep patterns...</h2>
            <div class="loader"></div>
        </div>
        <style>
            .loader {
                border: 16px solid #f3f3f3;
                border-top: 16px solid #3d5af1;
                border-radius: 50%;
                width: 120px;
                height: 120px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """
        display(HTML(loading_html))
        time.sleep(1.5)  # Dramatic pause
        clear_output(wait=True)

        # Prepare data for prediction
        X_pred = np.array([
            self.user_data["sleep_duration"],
            self.user_data["sleep_quality"],
            self.user_data["stress"]
        ]).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(X_pred)[0]
        probabilities = self.model.predict_proba(X_pred)[0]

        # Store results
        self.results = {
            "prediction": prediction,
            "probs": dict(zip(self.model.classes_, probabilities))
        }

        # Show results
        self.show_results()

    def show_results(self):
        """Display the final results with visualizations"""
        clear_output(wait=True)

        # Create dramatic reveal
        reveal_html = f"""
        <div style="text-align:center; margin:20px; animation: fadeIn 1s;">
            <h1 style="color:#3d5af1;">Your Sleep Mystery Result</h1>
            <div style="font-size:28px; margin:30px; padding:15px; border:3px solid #3d5af1;
                 border-radius:10px; display:inline-block;">
                {self.results["prediction"]}
            </div>
        </div>
        <style>
            @keyframes fadeIn {{
                0% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
        </style>
        """
        display(HTML(reveal_html))

        # Get sleep tips based on prediction
        tips = self.get_sleep_tips()

        # Create visualization with one regular subplot and one polar subplot
        fig = plt.figure(figsize=(12, 5))

        # Create bar chart subplot on the left
        ax1 = plt.subplot(1, 2, 1)

        # Bar chart of probabilities
        probs = self.results["probs"]
        classes = list(probs.keys())
        values = list(probs.values())
        colors = ['#3d5af1', '#22A39F', '#F2BE22', '#F24C3D']
        ax1.bar(classes, values, color=colors[:len(classes)])
        ax1.set_title('Sleep Condition Probabilities')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Probability')
        # Add percentage labels
        for i, v in enumerate(values):
            ax1.text(i, v + 0.05, f"{v:.0%}", ha='center')

        # Create polar subplot for radar chart on the right
        ax2 = plt.subplot(1, 2, 2, projection='polar')

        # Radar chart of user inputs
        categories = ['Sleep Duration', 'Sleep Quality', 'Stress Level']
        radar_values = [
            self.user_data["sleep_duration"] / 12,  # Normalized to 0-1
            self.user_data["sleep_quality"] / 10,
            self.user_data["stress"] / 10
        ]

        # Set up the angles for the radar chart (evenly spaced around the circle)
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()

        # Close the loops properly by adding the first element at the end
        radar_values = radar_values + [radar_values[0]]
        angles += angles[:1]  # Close the loop

        # Plot radar chart
        ax2.plot(angles, radar_values, 'o-', linewidth=2, color='#3d5af1')
        ax2.fill(angles, radar_values, alpha=0.25, color='#3d5af1')

        # Set category labels at the correct angles
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)

        # Set limits and title
        ax2.set_ylim(0, 1)
        ax2.set_title('Your Sleep Profile')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Display tips
        tips_html = f"""
        <div style="margin:20px; padding:15px; border:1px solid #ddd; border-radius:10px; background-color:#f9f9f9;">
            <h3 style="color:#3d5af1;">Sleep Tips for You:</h3>
            <ul style="list-style-type:none; padding:0;">
                {"".join([f'<li style="margin:8px 0; color:black;"><span style="color:#3d5af1;">•</span> {tip}</li>' for tip in tips])}
            </ul>
        </div>
        """
        display(HTML(tips_html))

        # Create restart button
        restart_button = widgets.Button(
            description='Try Again',
            button_style='success',
            icon='refresh',
            layout=widgets.Layout(width='40%', height='40px')
        )

        # Use the same pattern for the restart button
        def restart_handler(b):
            self.restart_game()
        restart_button.on_click(restart_handler)

        display(widgets.HBox([restart_button], layout=widgets.Layout(justify_content='center')))

        # Disclaimer
        disclaimer = """
        <div style="text-align:center; margin-top:20px; font-size:12px; color:#777;">
            This is a demonstration tool only. For actual sleep issues, consult a healthcare professional.
        </div>
        """
        display(HTML(disclaimer))

    def get_sleep_tips(self):
        """Return tips based on the predicted sleep condition"""
        prediction = self.results["prediction"]

        common_tips = [
            "Maintain a consistent sleep schedule",
            "Make your bedroom dark, quiet, and cool"
        ]

        if prediction == "Insomnia":
            specific_tips = [
                "Limit screen time 1 hour before bed",
                "Try relaxation techniques like deep breathing",
                "Avoid caffeine after noon"
            ]
        elif prediction == "Sleep Apnea":
            specific_tips = [
                "Sleep on your side rather than your back",
                "Maintain a healthy weight",
                "Consider talking to a doctor about a sleep study"
            ]
        elif prediction == "Hypersomnia":
            specific_tips = [
                "Set an alarm and avoid hitting snooze",
                "Get natural sunlight early in the day",
                "Establish a regular exercise routine"
            ]
        else:  # Normal Sleep
            specific_tips = [
                "Continue your good sleep habits",
                "Exercise regularly but not right before bed",
                "Limit long naps during the day"
            ]

        return common_tips + specific_tips

    def restart_game(self):
        """Restart the game from the beginning"""
        self.current_step = 0
        self.user_data = {}
        self.results = None
        self.start_game()

# Run the sleep mystery challenge
if __name__ == "__main__":
    # Install ipywidgets if not already installed
    try:
        import ipywidgets
    except ImportError:
        !pip install ipywidgets
        from google.colab import output
        output.clear()

    game = SleepMystery()
    game.start_game()
