
import streamlit as st
import joblib
import pandas as pd
import requests
import json

st.set_page_config(page_title="Smart House Price Predictor", page_icon="🏡", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

selected_columns = [
    'BedroomAbvGr', 'FullBath', 'KitchenAbvGr', 'PoolArea', 'LotArea',
    'YearBuilt', 'GarageCars', 'OverallQual', 'Neighborhood'
]

# Common neighborhood options (you can expand this list based on your dataset)
neighborhoods = [
    "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", 
    "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", 
    "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", 
    "Blmngtn", "BrDale", "SWISU", "Blueste"
]

st.title("🏡 Smart House Price Predictor")
st.markdown("Fill out the form below to get an AI-powered price prediction for your house")

# Create the main form
with st.form("house_prediction_form"):
    st.subheader("🏠 House Specifications")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📐 Basic Details**")
        bedrooms = st.number_input("Bedrooms Above Ground", min_value=0, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2, step=1)
        kitchens = st.number_input("Kitchens Above Ground", min_value=0, max_value=3, value=1, step=1)
        
    with col2:
        st.markdown("**📏 Size & Age**")
        lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=7000, step=500)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000, step=1)
        pool_area = st.number_input("Pool Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
        
    with col3:
        st.markdown("**🚗 Additional Features**")
        garage_cars = st.number_input("Garage Car Capacity", min_value=0, max_value=5, value=2, step=1)
        overall_quality = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6, step=1,
                                   help="1=Very Poor, 5=Average, 10=Excellent")
        neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict House Price", type="primary", use_container_width=True)

def get_gemini_insights(house_features, predicted_price):
    """Get detailed insights from Gemini AI about the house price prediction"""
    
    api_key = "AIzaSyBDHvuIaqxpJbvzDdeUexCLFlTzWdvUR08"
    
    if not api_key.strip():
        return "API key not available for AI insights."
    
    prompt = f"""
    Provide a professional real estate analysis for this house prediction:
    
    Predicted Price: ${predicted_price:,.2f}
    House Features: {dict(house_features.iloc[0])}
    
    Please include:
    1. Market analysis and price justification
    2. Investment insights and potential ROI considerations  
    3. How each feature impacts the house value
    4. Recommendations for the buyer/seller
    
    Keep the response comprehensive but concise (400-600 words).
    Focus on actionable insights and practical advice.
    """
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response_json = response.json()
        
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in response_json:
            return f"❌ Gemini API Error: {response_json['error'].get('message', 'Unknown error occurred.')}"
        else:
            return "❌ Unexpected response format from Gemini API."
            
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except Exception as e:
        return f"❌ Error getting AI insights: {str(e)}"

# Process form submission
if submitted:
    # Create input dataframe
    input_data = {
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "KitchenAbvGr": kitchens,
        "PoolArea": pool_area,
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "OverallQual": overall_quality,
        "Neighborhood": neighborhood
    }
    
    input_df = pd.DataFrame([input_data])[selected_columns]
    
    # Display input summary
    st.subheader("📋 House Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🛏️ Bedrooms", bedrooms)
        st.metric("🛁 Bathrooms", bathrooms)
    with col2:
        st.metric("🍳 Kitchens", kitchens)
        st.metric("🚗 Garage Cars", garage_cars)
    with col3:
        st.metric("📐 Lot Area", f"{lot_area:,} sq ft")
        st.metric("🏊 Pool Area", f"{pool_area} sq ft" if pool_area > 0 else "No Pool")
    with col4:
        st.metric("📅 Year Built", year_built)
        st.metric("⭐ Quality Rating", f"{overall_quality}/10")
    
    st.info(f"📍 **Neighborhood:** {neighborhood}")
    
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction
        st.subheader("💰 Price Prediction")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                <h2 style="margin: 0; color: white; font-size: 2.5em;">💰</h2>
                <h1 style="margin: 10px 0; color: white; font-size: 3em;">${prediction:,.0f}</h1>
                <p style="margin: 0; color: white; font-size: 1.2em;">Estimated House Price</p>
                <p style="margin: 5px 0; color: white; opacity: 0.9;">Model Accuracy: 83.43%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics
        st.subheader("📊 Additional Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_per_sqft = prediction / lot_area if lot_area > 0 else 0
            st.metric("💵 Price per Sq Ft", f"${price_per_sqft:.2f}")
        
        with col2:
            house_age = 2024 - year_built
            st.metric("🏠 House Age", f"{house_age} years")
        
        with col3:
            total_rooms = bedrooms + bathrooms
            st.metric("🏠 Total Rooms", f"{total_rooms}")
        
        with col4:
            has_pool = "Yes" if pool_area > 0 else "No"
            pool_color = "normal" if pool_area == 0 else "inverse"
            st.metric("🏊 Pool", has_pool)
        
        # Get AI insights
        with st.spinner("🤖 Generating AI analysis..."):
            insights = get_gemini_insights(input_df, prediction)
        
        st.subheader("🧠 AI-Powered Market Analysis")
        st.markdown(insights)
        
        # Price range estimation
        st.subheader("📈 Price Range Estimation")
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔻 Conservative Estimate", f"${lower_bound:,.0f}")
        with col2:
            st.metric("🎯 Most Likely Price", f"${prediction:,.0f}")
        with col3:
            st.metric("🔺 Optimistic Estimate", f"${upper_bound:,.0f}")
        
        st.info("💡 **Tip:** The actual selling price may vary based on market conditions, property condition, and negotiation factors.")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.error("Please check if the model file exists and is compatible with your scikit-learn version.")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🏡 Smart House Price Predictor | Built with Streamlit & AI</p>
    <p><em>Predictions are estimates based on historical data and should not be considered as professional appraisals.</em></p>
</div>
""", unsafe_allow_html=True)