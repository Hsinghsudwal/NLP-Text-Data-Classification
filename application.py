from flask import Flask, render_template, request
import joblib
from src.utility_file import Utility

app = Flask(__name__)

cv = joblib.load("models\count_vector.joblib")
model = joblib.load("models\grid_search_model.joblib")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        message = request.form["message"]

        data = [message]
        headlines = [Utility.clean_text(data)]

        y_pred1 = cv.transform(headlines)
        prediction = model.predict(y_pred1)[0]

        return render_template("index.html", data= prediction, text= message)


if __name__ == "__main__":
    app.run()
