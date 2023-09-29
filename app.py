from flask import Flask
from flask import render_template
from utils import get_month_predictions, get_2021_data, get_model_params

app = Flask(__name__)

@app.route("/")
def predict_2022():
    month_preds = get_month_predictions()
    data_2021 = get_2021_data()
    params =  get_model_params()
    return render_template('app.html', predictions=month_preds, data_2021=data_2021, params=params)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')