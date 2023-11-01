#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv('/opt/ml/processing/input/athletes_raw.csv')
data.shape

# Remove not relevant columns
data = data.dropna(subset=['region','age','weight','height','howlong','gender','eat', \
                               'train','background','experience','schedule','howlong', \
                               'deadlift','candj','snatch','backsq','experience',\
                               'background','schedule','howlong'])
data = data.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace',\
                              'filthy50','fgonebad','run400','run5k','pullups','train'])
# Clean Survey Data

decline_dict = {'Decline to answer|': np.nan}
data = data.replace(decline_dict)
data = data.dropna(subset=['background','experience','schedule','howlong','eat'])
data['gender'].value_counts()
data = data[data['gender'] != '--']

data['gender'].value_counts().plot(kind='bar')
data = pd.get_dummies(data, columns=['gender'], drop_first=True)
data['region'].value_counts().plot(kind='bar')
data = pd.get_dummies(data, columns=['region'], drop_first=True)

data['total_lift'] = data['candj'] + data['snatch'] + data['deadlift'] + data['backsq']

data.drop(['candj','snatch','deadlift','backsq'], axis=1, inplace=True)

# Creating binary columns for eat
data['eat_quality_foods'] = data['eat'].str.contains("I eat quality foods but don't measure the amount").astype(int)
data['eat_cheat_meals'] = data['eat'].str.contains("I eat 1-3 full cheat meals per week").astype(int)
data['eat_whatever_is_convenient'] = data['eat'].str.contains("I eat whatever is convenient").astype(int)
data['weigh_and_measure_food'] = data['eat'].str.contains("I weigh and measure my food").astype(int)
data['eat_strict_paleo'] = data['eat'].str.contains("I eat strict Paleo").astype(int)
data.drop('eat', axis=1, inplace=True)
# Creating binary columns for background
data['played_youth_highschool_sports'] = data['background'].str.contains("I played youth or high school level sports").astype(int)
data['regularly_play_recreational_sports'] = data['background'].str.contains("I regularly play recreational sports").astype(int)
data['no_athletic_background'] = data['background'].str.contains("I have no athletic background besides CrossFit").astype(int)
data['played_college_sports'] = data['background'].str.contains("I played college sports").astype(int)
data['played_professional_sports'] = data['background'].str.contains("I played professional sports").astype(int)
data.drop('background', axis=1, inplace=True)
# Creating binary columns for each category in the 'experience' feature

data['began_with_coach'] = data['experience'].str.contains("I began CrossFit with a coach \(e.g. at an affiliate\)").astype(int)
data['life_changing_experience'] = data['experience'].str.contains("I have had a life changing experience due to CrossFit").astype(int)
data['began_alone'] = data['experience'].str.contains("I began CrossFit by trying it alone \(without a coach\)").astype(int)
data['completed_level_1'] = data['experience'].str.contains("I have completed the CrossFit Level 1 certificate course").astype(int)
data['attended_specialty_courses'] = data['experience'].str.contains("I have attended one or more specialty courses").astype(int)
data['trains_others'] = data['experience'].str.contains("I train other people").astype(int)
data.drop('experience', axis=1, inplace=True)

# Creating binary columns for each category in the 'schedule' feature
data['1_workout_a_day'] = data['schedule'].str.contains("I usually only do 1 workout a day").astype(int)
data['multi_workouts_3plus_times'] = data['schedule'].str.contains("I do multiple workouts in a day 3+ times a week").astype(int)
data['multi_workouts_2x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 2x a week").astype(int)
data['multi_workouts_1x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 1x a week").astype(int)
data['strictly_schedule_rest'] = data['schedule'].str.contains("I strictly schedule my rest days").astype(int)
data['rest_4_or_more_days'] = data['schedule'].str.contains("I typically rest 4 or more days per month").astype(int)
data['rest_fewer_than_4_days'] = data['schedule'].str.contains("I typically rest fewer than 4 days per month").astype(int)
data.drop('schedule', axis=1, inplace=True)

# Creating binary columns for each category in the 'howlong' feature
data['2_4_years'] = data['howlong'].str.contains("2-4 years").astype(int)
data['1_2_years'] = data['howlong'].str.contains("1-2 years").astype(int)
data['4plus_years'] = data['howlong'].str.contains("4+ years").astype(int)
data['6_12_months'] = data['howlong'].str.contains("6-12 months").astype(int)
data['less_than_6_months'] = data['howlong'].str.contains("Less than 6 months").astype(int)
data.drop('howlong', axis=1, inplace=True)

data.to_csv('/opt/ml/processing/output/processed_data.csv', index=False)

