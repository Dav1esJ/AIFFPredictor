# AIFFPredictor
# AIFFPredictor is meant to look at past nfl data to make predictions on what players will score the best and help Fantasy Football players decide who to start and bench

What I have so far is only for assitance on drafting, the next step of this system would be on weekly predictions

#What works now and how effective it is

I created a Random Forest model that predicts how well players will do and a draft assistant that allows you to search for players and will recommend players from the model predictions based on what positions you need.  

# Results
<img width="678" height="451" alt="image" src="https://github.com/user-attachments/assets/cf17f8c5-e09a-407d-b058-8c5a2935c0d6" />

This graph shows that compared to the baseline the only model that performed better on both MAE and RMSE was the random forest model.  
The Random forest was approximately 4% better on MAE and approximately 7% better on RMSE, and that on average the random forest prediction is off by 1.97 points.  

<img width="537" height="547" alt="image" src="https://github.com/user-attachments/assets/da54cdc2-50bb-4b34-b9b4-98968982d05a" />

This graph shows that the majority of points are centered on lower values and those that score lower are easier to predict.  Because of this I recalculated values on more relevant players, players who score in their last 5 games an average of 8 points or more.

On Relevant players the baseline model has a MAE of 7.546 and an RMSE of 9.655 with a bias of -1.6, while the Random Forest model has a MAE of 6.83 and an RMSE of 8.90 with a bias of -0.124.  This shows that for relevant players the Random forest model has a MAE approximately 9.5% improvement and an improvement of 7.8% on RMSE.  This shows that on relevant players the random forest model on average predictions are off by 6.83 points Additionally the baseline predicts within 3 points of the actual score 25% of the time and the Random Forest is within 3 points 28% of the predictions.

This shows that there are some small improvements compared to the baseline model.

Additional steps I would like to take are designing some tests for the model and my code to ensure it is working properly

#Running this code
To run this code, first clone the repository and create a venv, then call pip install -r requirements.txt.
First run main.py and then call run_draft_assistant.py using the commands showing inside this file.
