import torch 
import torch.nn as nn
import datetime
import pandas as pd

DATE_START_IDX = 365

# define model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

# create model and load parameters
model = LinearRegression()
model.load_state_dict(torch.load('models/fetch_2022_linear.pth'))
model.eval() # switch to evaluation mode

# returns a list of all the dates in a given year
def get_all_dates_in_year(year):
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += datetime.timedelta(days=1)

    return all_dates

# given a date object, returns the date encoded as days after Jan 1 2021
def get_date_encoded(date):
    start_date = datetime.date(2021, 1, 1)
    end_date = date.date()

    days = 0
    current_date = start_date
    while current_date <= end_date:
        days += 1
        current_date += datetime.timedelta(days=1)

    return days-1

# gets predictions for receipts scanned in each month of 2022
def get_month_predictions():
    X_2022 = []
    y_2022 = []
    for i,d in enumerate(get_all_dates_in_year(2022)):
        X_2022.append(i + DATE_START_IDX)
        scans = model(torch.Tensor([i + DATE_START_IDX])).item()
        y_2022.append(scans)

    totals = [0] * 12
    curr_month = 1
    running = 0
    for i,d in enumerate(get_all_dates_in_year(2022)):
        if d.month != curr_month:
            totals[curr_month-1] = int(running)
            curr_month = d.month
            running = 0
        running += y_2022[i]
    totals[curr_month-1] = int(running)
    return totals

# gets monthly scans for each month in 2021
def get_2021_data():
    data = pd.read_csv('data_daily.csv', sep=',', header=None)
    daily_scans = [data.iloc[i,1] for i in range(len(data))]
    totals = [0] * 12
    curr_month = 1
    running = 0
    for i,d in enumerate(get_all_dates_in_year(2021)):
        if d.month != curr_month:
            totals[curr_month-1] = int(running)
            curr_month = d.month
            running = 0
        running += daily_scans[i]
    totals[curr_month-1] = int(running)
    return totals

# predicts receipts scanned for a given date
def predict(x):
    return model(torch.Tensor([x])).item()