import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
st.write("""

Weather Prediciton""")

st.sidebar.header("User Input")
def user_input():
  temperature=st.sidebar.slider('Temperature in Celcius',-10,49,20)
  app_temperature=st.sidebar.slider('Apparent Temperature (C)',-10,49,20)
  humidity=st.sidebar.slider('Humidity in %',0.01,1.0,0.51)
  wind_speed=st.sidebar.slider('Wind Speed in Km/h',5,80,16)
  wind_bearing=st.sidebar.slider('Wind bearing',0,360,235)
  visibility=st.sidebar.slider('Visibility',0,30,15)
  loud_cover=st.sidebar.slider('Loud Cover',0,0,0)
  pressure=st.sidebar.slider('Pressure(millibar)',998,1050,1001)
  data={'Temperature':temperature,
        'Apparent Temperature':app_temperature,
        'Humidty':humidity,
        'Wind Speed':wind_speed,
        'Wind Bearing':wind_bearing,
        'Visibility':visibility,
        'Loud Cover':loud_cover,
        'Pressure':pressure}
  f=pd.DataFrame(data,index=[0])
  return f
def user_input1():
    form=st.form(key='my_form')
    temperature=st.number_input(label='Temperature')
    humidity=st.number_input(label='Humidty')
    wind_speed=st.number_input(label='wind')
    visibility=st.number_input(label='visibility')
    pressure=st.number_input(label='pressure')
    data={'Temperature':35,
          
          'Humidty':0.51,
          'Wind Speed':16,
        
          'Visibility':10,
      
          'Pressure':1002}
    submit=form.form_submit_button('Submit')
    if submit:
		
      data={'Temperature':temperature,
          
          'Humidty':humidity,
          'Wind Speed':wind_speed,
        
          'Visibility':visibility,
      
          'Pressure':pressure}
    f=pd.DataFrame(data,index=[0])
    return f


def main():
  flag=0
  st.title("Upload File")
  csv_file=st.file_uploader("Upload",type=['csv','xslx'])
  
  if csv_file is not None:
    file_details = {"Filename":csv_file.name,"FileType":csv_file.type,"FileSize":csv_file.size}
    flag=1
    st.write(file_details)
    df=pd.read_csv(csv_file,parse_dates=['Formatted Date'])
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df = df.set_index('Formatted Date')

    le=preprocessing.LabelEncoder()
    df['Summary']=le.fit_transform(df['Daily Summary'])

    unique_summary=df['Summary'].unique()
    unique_daily_summary=df['Daily Summary'].unique()
    weather_map={unique_summary[i]:unique_daily_summary[i] for i in range(len(unique_summary))}

    summary=df.pop('Daily Summary')
    summary.fillna('Dilemma')

    data_columns = ['Summary','Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)']
    df_monthly_mean = df[data_columns].resample('MS').mean()
    df_monthly_mean.reset_index(inplace=True)

    summary1=pd.DataFrame(summary)

    col_y=df_monthly_mean[['Summary']]
    col_x_=df_monthly_mean[['Temperature (C)',  'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']]

    x_train,x_test,y_train,y_test=train_test_split(col_x_,col_y,test_size=0.20,random_state=True)

    regression = LinearRegression()
    regression.fit(x_train, y_train)

    pred_y = regression.predict(x_test)

    prediction = pd.DataFrame({'P summary': [k for k in pred_y],'Actual summ':[j for j in y_test['Summary']]})

    call=user_input1()
    st.subheader('Input Data')
    st.write(call)
    t=np.array(call.values)
    temp=[]
    for i in range(5):
      temp.append(t[0][i])
    g=regression.predict([temp])
    if (int(g)>max(unique_summary)):
      out=weather_map[max(unique_summary)]

    else:
      out=weather_map[int(g)]
    st.subheader("Prediction")
    st.write(out)

if __name__== '__main__':
  main()
