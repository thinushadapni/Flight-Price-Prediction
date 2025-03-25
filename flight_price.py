import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

data = pd.read_csv(r'C:\\Users\\N.Thinusha Dapni\\Desktop\\da_flight-20250325T052255Z-001\\da_flight\\Clean_Dataset.csv')

def get_user_input():
    st.sidebar.header("Enter Flight Details")
    source_city = st.sidebar.text_input("Source City (e.g., Delhi)")
    destination_city = st.sidebar.text_input("Destination City (e.g., Mumbai)")
    departure_time = st.sidebar.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening', 'Night', 'Any'])
    travel_class = st.sidebar.selectbox("Class", ['Economy', 'Business', 'Any'])
    
    if not source_city or not destination_city:
        st.warning("Please enter source and destination cities.")
        return None
    
    filtered_flights = data[
        (data['source_city'].str.lower().str.contains(source_city.lower())) &
        (data['destination_city'].str.lower().str.contains(destination_city.lower()))
    ]
    
    if departure_time != 'Any':
        filtered_flights = filtered_flights[filtered_flights['departure_time'].str.lower() == departure_time.lower()]
    
    if travel_class != 'Any':
        filtered_flights = filtered_flights[filtered_flights['class'].str.lower() == travel_class.lower()]
    
    if filtered_flights.empty:
        st.warning("No matching flights found. Try broadening your search criteria.")
        return None
    else:
        return filtered_flights, source_city, destination_city, travel_class

def convert_duration_to_minutes(duration_str):
    if isinstance(duration_str, str):
        hours, minutes = 0, 0
        if 'h' in duration_str:
            hours = int(duration_str.split('h')[0].strip())
            if 'm' in duration_str:
                minutes = int(duration_str.split('h')[1].split('m')[0].strip())
        elif 'm' in duration_str:
            minutes = int(duration_str.split('m')[0].strip())
        return hours * 60 + minutes
    return duration_str

def plot_graphs(filtered_flights, source_city, destination_city):
    if isinstance(filtered_flights['duration'].iloc[0], str) and ('h' in filtered_flights['duration'].iloc[0] or 'm' in filtered_flights['duration'].iloc[0]):
        filtered_flights['duration_minutes'] = filtered_flights['duration'].apply(convert_duration_to_minutes)
        duration_column = 'duration_minutes'
    else:
        duration_column = 'duration'
    
    with st.expander("Source City & Destination City"):
        st.subheader("Price Distribution for the Selected Route")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_flights['price'], kde=True, ax=ax)
        ax.set_title(f'Price Distribution: {source_city} ➔ {destination_city}')
        st.pyplot(fig)
        
        st.subheader("Average Price by Airline for the Route")
        fig, ax = plt.subplots(figsize=(12, 6))
        order = filtered_flights.groupby('airline')['price'].mean().sort_values(ascending=False).index
        sns.barplot(x='airline', y='price', data=filtered_flights, ax=ax, order=order)
        ax.set_title(f'Average Price by Airline: {source_city} ➔ {destination_city}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_order = [t for t in time_order if t in filtered_flights['departure_time'].unique()]
    
    with st.expander("Class (Economy/Business)"):
        st.subheader("Price Difference Between Economy and Business Class")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='class', y='price', data=filtered_flights, ax=ax)
        ax.set_title(f'Price by Class: {source_city} ➔ {destination_city}')
        st.pyplot(fig)
    
    with st.expander("Number of Stops"):
        st.subheader("Impact of Stops on Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='stops', y='price', data=filtered_flights, ax=ax)
        ax.set_title(f'Price by Number of Stops: {source_city} ➔ {destination_city}')
        st.pyplot(fig)
    
    with st.expander("Duration"):
        st.subheader("Duration vs. Price Relationship")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x=duration_column, y='price', hue='airline', data=filtered_flights, ax=ax)
        if duration_column == 'duration_minutes':
            ax.set_xlabel('Duration (minutes)')
        ax.set_title(f'Price vs Duration by Airline: {source_city} ➔ {destination_city}')
        plt.legend(title="Airline", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

def predict_price(flight_data):
    model = joblib.load(r"C:\\Users\\N.Thinusha Dapni\\Desktop\\da_flight-20250325T052255Z-001\\da_flight\\GradientBoosting_model.pkl")
    expected_columns = model.feature_names_in_
    input_vector = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    for feature, value in flight_data.items():
        encoded_col = f"{feature}_{value}"
        if encoded_col in expected_columns:
            input_vector[encoded_col] = 1
    
    predicted_price = model.predict(input_vector)[0]
    return round(predicted_price, 2)

def main():
    st.title("Flight Price Analysis & Prediction Dashboard")
    user_input = get_user_input()
    
    if user_input:
        filtered_flights, source_city, destination_city, travel_class = user_input
        st.write("### Matching Flights:")
        st.dataframe(filtered_flights[['airline', 'flight', 'departure_time', 'arrival_time', 'stops', 'duration', 'price']].drop_duplicates())
        plot_graphs(filtered_flights, source_city, destination_city)
    
        flight_features = {
            'source_city': source_city,
            'destination_city': destination_city,
            'class': travel_class,
            'stops': filtered_flights['stops'].mode()[0]
        }
        
        predicted_price = predict_price(flight_features)
        st.success(f"Predicted Price: ${predicted_price}")

if __name__ == "__main__":
    main()
