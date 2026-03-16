#importing libraries
from datetime import datetime
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # plotly for creating interactive plots
import requests



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

plt.style.use('dark_background')

np.random.seed(42)

# get API URL, default to localhost for dev
API_URL = os.getenv("API_URL", "http://localhost:8000")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STYLE_PATH =  "./style.css"

def get_forecast():
    response = requests.get(f"{API_URL}/forecast/latest")
    return response.json()

def get_historical_data():
    response = requests.get(f"{API_URL}/history")
    return response.json()

def get_metrics():
    response = requests.get(f"{API_URL}/metrics")
    return response.json()

@st.cache_data(ttl='1h') # cache data for 1 hour to reduce API calls
def get_cached_data():
    return get_forecast(), get_historical_data(), get_metrics()


def get_pct_change(ref_date,forecast_df,historical_df,months=1):
    # ensure ref_date is in datetime format from 'YYYY-MM' string
    
    ref_date = pd.to_datetime(ref_date, format='%Y-%m')
    ref_in_historical = ref_date <= historical_df['Date'].max()
    previous_date = ref_date - pd.DateOffset(months=months)
    prev_in_historical = previous_date <= historical_df['Date'].max()

    #st.write(f"Reference date ({ref_date.strftime('%Y-%m')}) in historical data: {ref_in_historical}")
    #st.write(f"Previous date ({previous_date.strftime('%Y-%m')}) in historical data: {prev_in_historical}")


    if not prev_in_historical:
        #st.dataframe(forecast_df)
        ref_df = forecast_df.loc[forecast_df['Date'].astype(str).str.contains(ref_date.strftime('%Y-%m'))].reset_index(drop=True)
        ordered_ref_value = ref_df.sort_values('Category')['Value']
        prev_df = forecast_df.loc[forecast_df['Date'].astype(str).str.contains(previous_date.strftime('%Y-%m'))].reset_index(drop=True)
        ordered_prev_value = prev_df.sort_values('Category')['Value']
    
    elif ref_in_historical and prev_in_historical:
        ref_df = historical_df.loc[historical_df['Date'] == ref_date]
        ordered_ref_value = ref_df.sort_values('Category')['Value']
        prev_df = historical_df.loc[historical_df['Date'] == previous_date]
        ordered_prev_value = prev_df.sort_values('Category')['Value']        
        
    elif (not ref_in_historical) and prev_in_historical:
        #st.dataframe(forecast_df)
        #get the forecasted value for ref_month
        ref_df = forecast_df.loc[forecast_df['Date'].astype(str).str.contains(ref_date.strftime('%Y-%m'))].reset_index(drop=True)
        ordered_ref_value = ref_df.sort_values('Category')['Value']
        prev_df = historical_df.loc[historical_df['Date'] == previous_date].reset_index(drop=True)
        ordered_prev_value = prev_df.sort_values('Category')['Value']

    

    if not ref_in_historical:
        ref_date = f"{ref_date.strftime('%B,%Y')} (Forecast)"
    else:
        ref_date = ref_date.strftime('%B,%Y')
    if not prev_in_historical:
        previous_date = f"{previous_date.strftime('%B,%Y')} (Forecast)"
    else:
        previous_date = previous_date.strftime('%B,%Y')

    #st.write("displaying reference and previous dataframes used for percentage change calculation:")
    #st.dataframe(ref_df)
    #st.dataframe(prev_df)

    df = pd.DataFrame({'Category': ref_df.sort_values('Category')['Category'],
                        f'{previous_date} - {ref_date} % change': (ordered_ref_value.values-ordered_prev_value.values)/ordered_prev_value.values * 100})
    return df.reset_index(drop=True)

def combine_forecast_with_historical(forecast_df, historical_df):
    '''
    Combines forecast and historical data for plotting.
    '''
    # new column to indicate whether it's historical or forecast
    forecast_df['Data Type'] = 'Forecast'
    historical_df['Data Type'] = 'Historical'
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    return combined_df.sort_values('Date').reset_index(drop=True)

def main():
    '''
    Main function to run the Streamlit app.
    '''

    current_forecast, historical_data, model_metrics = get_cached_data()

    hist_df = pd.DataFrame(historical_data)
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])

    current_forecast = pd.DataFrame(current_forecast)

    combo = combine_forecast_with_historical(current_forecast, hist_df)

    st.title('South Africa CPI Nowcasting Application')
    #image = STATIC_DIR / "background_image.jpg"


    #with open(STYLE_PATH, encoding='utf-8') as f:
    #   st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


    tab_home, tab_visuals, tab_model_health = st.tabs(['Home','CPI Visualizations',
                                                       'Model Health & Metrics'])
        
    with tab_home:
        st.write('The predictions of this web application are powered by a Catboost\
                Regressor model trained each on a time-series \
                of historic CPI values plus carefully engineered predictive features.\
                The model is retrained every month as new CPI data becomes available.')
        st.write('The data used to train the model is ingested from the [Stats SA website](https://www.statssa.gov.za/).')
        
        col1, col2, col3 = st.columns((1,1,1),gap='medium')

        # convert last train date to more readable format
        train_date = datetime.strptime(model_metrics['last_train_date'],
                                        '%Y%m%d_%H%M').strftime('%d %B %Y')
        model_metrics['rmse'] = round(model_metrics['rmse'], 2)
        col1.metric("Last Train Date", train_date)
        col2.metric("Model RMSE", model_metrics['rmse'])
        col3.metric("CPI Last Published for", hist_df['Date'].max().strftime('%B %Y'))

        st.write('The table below shows the latest CPI forecast generated by the model:')
        disp_current_forecast = current_forecast.drop(columns=['Data Type'])
        # convert date to y-m format
        disp_current_forecast['Date'] = pd.to_datetime(
            disp_current_forecast['Date']).dt.strftime('%Y-%m')
        disp_current_forecast = disp_current_forecast.rename(columns={'Date': 'Forecast Date',
                                                             'Value': 'Predicted CPI Value'})
        disp_current_forecast = disp_current_forecast.sort_values(
            by='Forecast Date', ascending=False).reset_index(drop=True)
        st.dataframe(disp_current_forecast)

        # option to download the forecast as a CSV
        csv = disp_current_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast as CSV",
                           data=csv, file_name="cpi_forecast.csv",
                            mime="text/csv")
    
    with tab_visuals:

        st.write('The plot below shows the historical CPI values along with the latest forecast:')
        
        unique_categories = hist_df['Category'].unique()
        cat_options =st.multiselect('Category:',unique_categories,default=unique_categories)

        hist_df['y-m'] = hist_df['Date'].dt.to_period('M')
        num_months = st.slider('Number of Months:',
                                min_value=6,
                                max_value=hist_df['y-m'].nunique(),
                                step=1,value=12)
        
        # filter the dataframe based on selected categories and number of months
        filtered_df = combo[combo['Category'].isin(cat_options)]
        
        latest_date = filtered_df['Date'].max()
        filtered_df = filtered_df[filtered_df['Date'] >= latest_date - pd.DateOffset(months=num_months)]

        # plot using plotly express for interactivity plus distinguish
        #  forecast vs historical with line style
        fig = px.line(filtered_df, x='Date', y='Value', color='Category',
                       title='Historical CPI with Latest Forecast',
                       line_dash='Data Type')
        st.plotly_chart(fig, width='stretch')

        # get percentage change defaulting to a reference date of the latest forecast month
        
        ref_date = pd.to_datetime(current_forecast['Date'].max())
        ref_date_ch = st.date_input('Reference Date for % change:',
                                  value=ref_date, min_value=filtered_df['Date'].min(),
                                    max_value=current_forecast['Date'].max())
        horizon = st.selectbox('Horizon (in months) for % comparison:', options=[1,3,6,12],
                                index=0,
                                help="Number of months to look back for percentage change calculation")
        pct_change_df = get_pct_change(ref_date_ch.strftime('%Y-%m'),
                                        current_forecast, hist_df, months=horizon)

        #st.dataframe(pct_change_df)
        # display percentage change with a plotly express bar chart
        fig2 = px.bar(pct_change_df, x='Category', y=pct_change_df.columns[1], 
                      title=pct_change_df.columns[1],
                        labels={pct_change_df.columns[1]: 'Percentage Change (%)'})
        st.plotly_chart(fig2, width='stretch')

    with tab_model_health:
        features_importance = model_metrics.get('features_importance', {})
        if features_importance:
            fi_df = pd.DataFrame({'Feature': list(features_importance.keys()),
                                  'Importance': list(features_importance.values())})
            # more descriptive feature names mapping
            feature_name_mapping = {name:name.replace(
                'Value_','CPI Lag ') for name in features_importance.keys()}
            fi_df = fi_df.sort_values(by='Importance', ascending=False)
            fi_df['Feature'] = fi_df['Feature'].map(feature_name_mapping)
            fig3 = px.bar(fi_df.head(5), x='Feature', y='Importance', title='Top 5 Feature Importance')
            st.plotly_chart(fig3, width='stretch')


    st.markdown('''
    ---
    Created by [Isaac Oluwafemi Ogunniyi](https://linkedin.com/in/isaac-oluwafemi-ogunniyi)
    ''')

if __name__ == '__main__':
    main()
    