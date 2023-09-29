from flask import Flask
from flask import render_template
from utils import get_month_predictions, get_2021_data, get_date_encoded, predict
import datetime

app = Flask(__name__)

# main app page
@app.route("/")
def predict_2022():
    month_preds = get_month_predictions()
    data_2021 = get_2021_data()
    return render_template('app.html', predictions=month_preds, data_2021=data_2021)

# endpoint to predict scans for a given date
@app.route("/predict/<date>")
def predict_date(date):
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    encoded_date = get_date_encoded(date)
    result = predict(encoded_date)
    return f'{result}'

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')