import seaborn as sns
import pandas as pd
import streamlit as st
import openai
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import easyocr
import os
import datetime

# Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")  # Correct
DATA_FILE = "water_quality_data.csv"
REGULATORY_STANDARDS = {
    "pH": (6.5, 8.5),
    "Chlorine (mg/L)": (0, 4),
    "Hardness (mg/L as CaCO3)": (0, 120),
    "Nitrates (mg/L)": (0, 10),
    "Lead (µg/L)": (0, 15),
}

# Functions
def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Sends a prompt to the OpenAI API and retrieves a response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[ 
                {"role": "system", "content": "You are a water quality expert offering simple suggestions and analysis. Give simple suggestion that educate citizens"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def load_data():
    if os.path.isfile(DATA_FILE):
        # Load data and ensure valid datetime format for Date column
        data = pd.read_csv(DATA_FILE, dtype={"Zipcode": str})
        data.rename(columns={"zipcode": "Zipcode"}, inplace=True)  # Ensure column consistency
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.date  # Ensure consistent date format
        invalid_date_rows = data[data["Date"].isna()]
        if not invalid_date_rows.empty:
            st.warning("Some rows have invalid dates and were excluded.")
            data = data.dropna(subset=["Date"])  # Drop invalid dates
        return data
    return pd.DataFrame(columns=["Zipcode", "Date", "pH", "Chlorine (mg/L)", 
                                  "Hardness (mg/L as CaCO3)", "Nitrates (mg/L)", "Lead (µg/L)", "Notes"])

def save_data(data):
    data.to_csv(DATA_FILE, index=False)

def specific_adjustment(param):
    """Return a specific adjustment recommendation based on the parameter."""
    adjustments = {
        "pH": "adding an alkaline or acidic agent to bring pH within 6.5-8.5 range.",
        "Chlorine (mg/L)": "adjusting chlorine dosing to remain within the 0-4 mg/L range for safe disinfection.",
        "Hardness (mg/L as CaCO3)": "installing a water softener if levels are too high, or adding calcium if too low.",
        "Nitrates (mg/L)": "reducing nitrate contamination sources like fertilizers, or improving water filtration.",
        "Lead (µg/L)": "replacing lead-containing pipes or fittings, and using lead-removal filters."
    }
    return adjustments.get(param, "consulting water quality professionals for appropriate adjustments.")

def get_zipcode_from_coordinates(lat, lon):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("address", {}).get("postcode", "Unknown ZIP Code")
    except Exception as e:
        st.error(f"Error retrieving ZIP code: {e}")
        return "Unknown ZIP Code"

def initialize_reader():
    return easyocr.Reader(['en'])

def extract_text_from_image(image):
    """
    Extracts text from an uploaded image using EasyOCR.
    """
    reader = initialize_reader()
    img = np.array(image)
    results = reader.readtext(img)
    return " ".join([text for _, text, _ in results])

def compare_zipcodes(data, selected_zipcodes, parameters): 
    """
    Compare water quality across selected zip codes and get AI analysis.
    """
    if len(selected_zipcodes) < 2:
        st.warning("Please select at least two zip codes for comparison.")
        return

    # Filter data for the selected zip codes
    filtered_data = data[data["Zipcode"].isin(selected_zipcodes)]
    if filtered_data.empty:
        st.warning("No data available for the selected zip codes.")
        return

    # Aggregate data by Zipcode and Parameter
    aggregated = filtered_data.groupby("Zipcode")[parameters].mean()
    
    # Display aggregated data
    st.write("### Aggregated Data Across Zipcodes")
    st.dataframe(aggregated)

    # Add Heatmap
    st.write("### Heatmap of Water Quality Metrics")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        aggregated,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Value'},
        linewidths=0.5,
    )
    plt.title("Heatmap of Water Quality Metrics Across Selected Zip Codes", fontsize=14)
    plt.xlabel("Parameter", fontsize=12)
    plt.ylabel("Zip Code", fontsize=12)
    plt.tight_layout()  # Adjust layout to avoid overlap
    st.pyplot(fig)

    # Generate OpenAI prompt
    prompt = (
        "Analyze and compare the following water quality metrics across zip codes. "
        "Provide insights on which zip codes perform better or worse for each parameter. "
        "Here is the data:\n\n"
        f"{aggregated.to_csv(index=True)}"
    )

    # Get analysis from OpenAI
    st.write("### AI Analysis of Water Quality Across Zip Codes")
    try:
        analysis = get_completion(prompt)
        st.write(analysis)
    except Exception as e:
        st.error(f"Failed to get AI analysis: {e}")

# Streamlit App
st.set_page_config(page_title="TrueBlue Water Quality Management", page_icon=":droplet:", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');


    /* General body styles */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;  /* Light neutral background */
        color: #343a40;  /* Dark grey text */
    }


    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e3f2fd;  /* Light blue sidebar */
        padding-top: 20px;
    }


    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #0077cc;  /* Logo's primary blue */
    }


    /* Buttons and inputs */
    .stButton>button {
        background: linear-gradient(90deg, #0077cc, #85d7ff);
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }


    .stButton>button:hover {
        background: #3fa9f5;
        color: white;
    }


    input, select, textarea {
        border: 2px solid #0077cc;
        border-radius: 8px;
    }


    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
    }


    /* Footer */
    footer {
        text-align: center;
        font-size: 14px;
        padding: 10px;
        background: #e3f2fd;
        border-top: 1px solid #0077cc;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar selection for features
st.sidebar.title("Navigation")
options = ["Simple Feature", "Professional Feature"]
selected_option = st.sidebar.radio("Select a Feature", options)  # This is where selected_option is defined.


# Load the logo
logo_path = "TrueBlue.png"  # Ensure this is the correct path to your logo image
try:
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        # No logo on sidebar
        st.sidebar.empty()  # Remove the sidebar logo
    else:
        st.sidebar.markdown("<p style='text-align: center; color: #6c757d;'>Logo not found</p>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error("An error occurred while loading the logo.")
    st.sidebar.write(str(e))

# Page title with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_image, width=400)  # Display logo (larger for the main page)
with col2:
    st.markdown("<h1 style='font-family:Poppins; color:#0077cc;'> Water Quality Management</h1>", unsafe_allow_html=True)

# Load data
data = load_data()

# Simple Feature
if selected_option == "Simple Feature":
    st.subheader("Citizen Water Quality Input & Suggestions")
    input_option = st.radio("Select Input Method:", ("Manual Input", "Image Upload"))

    st.markdown("### Location Selection Using the Interactive Map")
    st.markdown("Click on the map to select your location. The ZIP code will be automatically detected.")

    # Interactive map with ZIP Code input below
    map_center = [37.7749, -122.4194]
    folium_map = folium.Map(location=map_center, zoom_start=10)
    folium.LatLngPopup().add_to(folium_map)

    map_data = st_folium(folium_map, width=700, height=500)  # Map display

    detected_zipcode = None
    if map_data and map_data.get("last_clicked"):
        last_clicked = map_data["last_clicked"]
        if last_clicked:
            lat, lon = last_clicked["lat"], last_clicked["lng"]
            detected_zipcode = get_zipcode_from_coordinates(lat, lon)
            st.success(f"Detected ZIP Code: {detected_zipcode}")
    else:
        st.info("Click on the map to select a location and detect a ZIP code.")

    # ZIP Code input field below the map
    confirmed_zipcode = st.text_input("Confirm or Update ZIP Code:", value=detected_zipcode or "")

    # Manual Input
    if input_option == "Manual Input":
        date = st.date_input("Enter Date:", value=datetime.date.today())  # Default to today's date
        readings = {
            "pH": st.number_input("Enter pH level:", 0.0, 14.0, step=0.1),
            "Chlorine (mg/L)": st.number_input("Enter Chlorine (mg/L):", 0.0, 10.0, step=0.1),
            "Hardness (mg/L as CaCO3)": st.number_input("Enter Hardness (mg/L):", 0.0, 500.0, step=1.0),
            "Nitrates (mg/L)": st.number_input("Enter Nitrates (mg/L):", 0.0, 50.0, step=0.1),
            "Lead (µg/L)": st.number_input("Enter Lead (µg/L):", 0.0, 100.0, step=0.1)
        }

        if st.button("Submit"):
            # Create new entry with user data
            new_entry = {"Zipcode": detected_zipcode, "Date": date, **readings, "Notes": ""}
            data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
            save_data(data)
            st.success("Data submitted successfully!")
            
            # Generate Suggestions based on the input data
            try:
                prompt = f"Water quality data: {new_entry}. Provide simple solutions to improve water quality."
                suggestions = get_completion(prompt)
                st.write("### Suggestions to Improve Water Quality")
                st.write(suggestions)
            except Exception as e:
                st.error(f"Failed to generate suggestions: {e}")
    
    # Image Upload Input
    elif input_option == "Image Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
        zipcode = st.text_input("Enter Zipcode for this entry:", value=confirmed_zipcode)
        if uploaded_file and zipcode:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            extracted_text = extract_text_from_image(image)  # Function to extract text from image
            st.write("### Extracted Text")
            st.write(extracted_text)
            
            # Create new entry based on extracted data
            new_entry = {
                "Zipcode": zipcode, 
                "Date": pd.Timestamp.now().date(), 
                "pH": None, "Chlorine (mg/L)": None, "Hardness (mg/L as CaCO3)": None, 
                "Nitrates (mg/L)": None, "Lead (µg/L)": None, "Notes": extracted_text
            }
            data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
            save_data(data)
            st.success("Data and extracted text saved successfully!")
            
            # Generate Suggestions based on extracted text
            try:
                prompt = f"Extracted water quality details: {extracted_text}. Provide simple solutions to improve water quality for Zipcode {zipcode}."
                suggestions = get_completion(prompt)
                st.write("### Suggestions to Improve Water Quality")
                st.write(suggestions)
            except Exception as e:
                st.error(f"Failed to generate suggestions: {e}")

# Main Streamlit Application for Professional Feature
elif selected_option == "Professional Feature":
    st.subheader("Water Quality Dashboard for Professionals")

    # Filter Data by Zipcode
    selected_zipcodes = st.multiselect(
        "Select Zipcodes for Analysis",
        options=data["Zipcode"].unique(),
        help="Choose the zip codes to analyze water quality data."
    )
    
    # Apply the filter
    filtered_data = data[data["Zipcode"].isin(selected_zipcodes)] if selected_zipcodes else data.copy()

    st.write("### Filtered Data:")
    st.dataframe(filtered_data)

    # Displaying Historical Trends
    st.write("### Historical Trends")
    param_to_show = st.selectbox(
        "Select the parameter to view the trend for:",
        options=list(REGULATORY_STANDARDS.keys())
    )

    if param_to_show in filtered_data.columns:
        # Create a plot for historical trends of the selected parameter
        plt.figure(figsize=(10, 4))
        for zipcode in selected_zipcodes:
            zip_data = filtered_data[filtered_data["Zipcode"] == zipcode]
            if not zip_data.empty:
                zip_data = zip_data.sort_values(by="Date")
                plt.plot(
                    zip_data["Date"], 
                    zip_data[param_to_show], 
                    label=f"Zipcode {zipcode}", 
                    linewidth=2
                )
        
        # Draw regulatory standards lines
        min_val, max_val = REGULATORY_STANDARDS[param_to_show]
        plt.axhline(min_val, color="green", linestyle="--", label="Min Standard")
        plt.axhline(max_val, color="red", linestyle="--", label="Max Standard")
        
        # Formatting the plot
        plt.title(f"{param_to_show} Trends Over Time")
        plt.xlabel("Date")
        plt.ylabel(param_to_show)
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.warning(f"No data available for the selected parameter: {param_to_show}")

    # Edit and Delete functionality for Existing Data
    st.write("### Edit and Delete Existing Data")

    for idx, row in filtered_data.iterrows():
        with st.expander(f"Entry {idx+1} - Zipcode: {row['Zipcode']} | Date: {row['Date']}"):
            # Input for editing notes
            new_notes = st.text_input("Edit Notes", value=row['Notes'], key=f"edit_notes_{idx}")

            # Edit button logic
            if st.button("Save Changes", key=f"save_{idx}"):
                data.at[idx, 'Notes'] = new_notes
                save_data(data)
                st.success(f"Entry {idx+1} updated successfully!")

            # Delete button logic
            if st.button("Delete", key=f"delete_{idx}"):
                data = data.drop(idx)
                save_data(data)
                st.success(f"Entry {idx+1} deleted successfully!")

   # Interactive Map with Markers
    st.subheader("Interactive Map with Markers")

    if not filtered_data.empty:
        # Known zip code coordinates with fallback
        known_zipcode_coords = {
            "95110": (37.3422, -121.8996),
            "95112": (37.3535, -121.8865),
            "95113": (37.3333, -121.8907),
            "95116": (37.3496, -121.8569),
            "95117": (37.3126, -121.9502),
            "95118": (37.2505, -121.8891),
            "95120": (37.2060, -121.8133),
        }

        map_center = [37.3382, -121.8863]  # Default to San Jose
        folium_map = folium.Map(location=map_center, zoom_start=12)

        def get_coordinates_for_zipcode(zipcode):
            """Retrieve coordinates for a given zip code."""
            if zipcode in known_zipcode_coords:
                return known_zipcode_coords[zipcode]
            else:
                try:
                    response = requests.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={"postalcode": zipcode, "format": "json", "country": "US"},
                        timeout=10
                    )
                    response.raise_for_status()
                    results = response.json()
                    if results:
                        return float(results[0]["lat"]), float(results[0]["lon"])
                except Exception as e:
                    st.warning(f"Unable to determine coordinates for ZIP code {zipcode}. Using default map center.")
            return map_center

        # Add markers for each entry in the filtered data
        for _, row in filtered_data.iterrows():
            zipcode = row["Zipcode"]
            lat, lon = get_coordinates_for_zipcode(zipcode)
            popup_info = (
                f"<b>Zipcode:</b> {zipcode}<br>"
                f"<b>pH:</b> {row.get('pH', 'N/A')}<br>"
                f"<b>Chlorine (mg/L):</b> {row.get('Chlorine (mg/L)', 'N/A')}<br>"
                f"<b>Hardness (mg/L as CaCO3):</b> {row.get('Hardness (mg/L as CaCO3)', 'N/A')}<br>"
                f"<b>Nitrates (mg/L):</b> {row.get('Nitrates (mg/L)', 'N/A')}<br>"
                f"<b>Lead (µg/L):</b> {row.get('Lead (µg/L)', 'N/A')}<br>"
                f"<b>Notes:</b> {row.get('Notes', 'N/A')}"
            )
            folium.Marker(
                location=[lat, lon],
                popup=popup_info,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(folium_map)

        # Display the map
        st_folium(folium_map, width=700, height=500)
    else:
        st.warning("No data available to display on the map.")

# Compare Water Quality Across Zip Codes
    st.markdown("### Compare Water Quality Across Zip Codes")
    # Step 1: User selects zip codes and parameters
    comparison_zipcodes = st.multiselect(
        "Select Zip Codes to Compare",
        options=data["Zipcode"].unique(),
        help="Select at least two zip codes to compare water quality metrics."
    )

    parameters_to_compare = st.multiselect(
        "Select Parameters for Comparison",
        options=[param for param in REGULATORY_STANDARDS.keys()],
        default=list(REGULATORY_STANDARDS.keys()),
        help="Select the water quality parameters to include in the comparison."
    )

    # Step 2: Trigger comparison when the user clicks the button
    if st.button("Compare Zip Codes"):
        compare_zipcodes(data, comparison_zipcodes, parameters_to_compare)
