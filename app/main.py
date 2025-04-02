# Import necessary libraries
import streamlit as st              
import pickle as pickle            
import pandas as pd               
import numpy as np                
import plotly.graph_objects as go   

# Read the dataset, remove unwanted columns, and return cleaned dataset
def clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data    

# Add a sidebar in the app for user input
def add_sidebar():
    # Set sidebar title and clean the data
    st.sidebar.header("Cell Nuclei Details")
    data = clean_data()

    # List of features with their corresponding dataset column names
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Dictionary to store user input values
    input_dict = {}

    # Create sliders for each feature to allow user input
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict
 
# Normalize user input based on the dataset's min-max values
def get_scaled_values(input_dict):
    data = clean_data()

    X = data.drop(['diagnosis'], axis=1) 
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

# Generate a radar chart to visualize input values
def get_radar_chart(input_data): 
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = [
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Worst Value'
    ))

    fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
    )

    return fig

# Load the model and scaler, make predictions, and display results
def add_predictions(input_data):
    # Load the model and scaler created in the model directory
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Convert input dictionary values to a NumPy array and reshape it for model input
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    # Make a prediction using the trained model
    prediction = model.predict(input_array_scaled)

    # Get probability estimates for both benign and malignant classes
    benign_prob = model.predict_proba(input_array_scaled)[0][0]
    malignant_prob = model.predict_proba(input_array_scaled)[0][1]
    
    # Display prediction results with custom styling
    st.subheader("Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>BENIGN</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>MALIGNANT</span>", unsafe_allow_html=True)
    
    st.write(f"Benignness probability: {benign_prob}")
    st.write(f"Malignancy probability: {malignant_prob}")
    st.write("DISCLAIMER: This app assists medical professionals in making a diagnosis. It should not be used as a substitute for a professional diagnosis.")

def main():
    # Configure app page layout 
    st.set_page_config(
        page_title = "Breast Cancer Predictor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load CSS style
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    # Collect user input from the sidebar
    input_data = add_sidebar()
    
    # Create a container for the main content
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app uses a machine learning model to predict whether a breast mass is benign or malignant based on the input details of the cell nuclei in the sidebar.")

    # Create two columns to display radar chart and prediction results
    col1, col2 = st.columns([4, 1])

    # Display the radar chart
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    # Display the predidtion results
    with col2:
        add_predictions(input_data)

# Ensure the script runs only when executed directly 
if __name__ == '__main__':
    main()