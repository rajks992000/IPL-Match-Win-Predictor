import streamlit as st
import sklearn
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')



# declaring teams

Teams=['Sunrisers Hyderabad',
      'Mumbai Indians',
      'Royal Challengers Bangalore',
      'Kolkata Knight Riders',
      'Delhi Capitals',
      'Kings XI Punjab',
      'Chennai Super Kings',
      'Rajasthan Royals',
      ]

# declaring cities

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Cuttack', 'Pune', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Raipur', 'Mohali', 'Bengaluru']


pipe=pickle.load(open('pipe.pkl','rb'))

st.title('IPL WIN PREDICTOR')

col1,col2=st.columns(2)

with col1:
    battingteam=st.selectbox('Select Batting Team',sorted(Teams))

with col2:
    bowlingteam=st.selectbox('Select Bowling Team',sorted(Teams))

city=st.selectbox('Select the city where the match is being Played',sorted(cities))

target=st.number_input('Target')

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score')

with col4:
    overs=st.number_input('Overs Completed')

with col5:
    wickets=st.number_input('Wickets Fallen')

if st.button('Predict Probability'):

    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    cur_run_rate=score/overs
    req_run_rate=(runs_left*6)/balls_left

    data={
        'batting_team':[battingteam],
        'bowling_team':[bowlingteam],
        'city':[city],
        'total_runs_x':[target],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets],
        'cur_run_rate':[cur_run_rate],
        'required_run_rate':[req_run_rate],

    }
    input_df=pd.DataFrame(data)

    result=pipe.predict_proba(input_df)

    losspro=result[0][0]
    winpro=result[0][1]

    st.header('Batting Team Chance:-'+str(round(winpro*100))+'%')
    st.header('Bowling Team Chance:-'+str(round(winpro*100))+'%')

