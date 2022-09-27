import streamlit as st
import pickle
import pyspark
st.title('IPL Win Predictor')
cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
       'Visakhapatnam', 'Bengaluru', 'Jaipur', 'Bangalore', 'Raipur',
       'Ranchi', 'Cuttack', 'Nagpur', 'Johannesburg', 'Centurion',
       'Durban', 'Bloemfontein', 'Port Elizabeth', 'Kimberley',
       'East London', 'Cape Town']
teams=['Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Chennai Super Kings',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Mumbai Indians']

pipe=pickle.load(open('spark_pickle.pkl', 'rb'))

col1,col2=st.columns(2)

with col1:
    BattingTeam=st.selectbox('Select the batting team',sorted(teams))
with col2:
    BowlingTeam=st.selectbox('Select the bowling team',sorted(teams))

selected_city=st.selectbox('Select the venue',sorted(cities))

target =st.number_input('Target')

col3,col4,col5=st.columns(3)

with col3:
    score=int(st.number_input('Score'))
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('Wickets out')
if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets= 10-wickets
    crr=score/overs
    rrr= (runs_left*6)/balls_left

    input_df = pd.DataFrame({'BattingTeam':[BattingTeam], 'BowlingTeam':[BowlingTeam], 'City':[selected_city], 'runs_left':[runs_left], 'balls_left':[balls_left],
       'wickets':[wickets], 'total_run_x':[target], 'crr':[crr], 'rrr':[rrr]})

    st.table(input_df)
    result= pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(BattingTeam + "- " + str(round(win * 100)) + "%")
    st.header(BowlingTeam + "- " + str(round(loss * 100)) + "%")