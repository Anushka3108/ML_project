from flask import Flask, request, render_template
from ML_project.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        try:
            # ✅ Get values from form
            reading = float(request.form.get('reading_score'))
            writing = float(request.form.get('writing_score'))

            # ✅ Backend validation (VERY IMPORTANT)
            if not (0 <= reading <= 100 and 0 <= writing <= 100):
                return render_template('index.html', error="Scores must be between 0 and 100")

            # ✅ Create custom data object
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading,
                writing_score=writing
            )

            # ✅ Convert to dataframe
            pred_df = data.get_data_as_data_frame()

            # ✅ Prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(pred_df)

            # ✅ Fix: ensure proper output
            result = float(round(prediction[0], 2))

            # ✅ Chart data
            features = ['Reading Score', 'Writing Score']
            values = [reading, writing]

            return render_template(
                "index.html",
                results=result,
                features=features,
                values=values
            )

        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
    