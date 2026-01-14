from flask import Flask, request, render_template
from src.Pipeline.Prediction_pipeline import PredictionPipeline
from src.Pipeline.custom_data import CustomData

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = CustomData(
            track_name=request.form.get("track_name"),
            artist_name=request.form.get("artist_name"),
            artist_genres=request.form.get("artist_genres"),
            album_name=request.form.get("album_name"),
            track_popularity=float(request.form.get("track_popularity")),
            artist_popularity=float(request.form.get("artist_popularity")),
            artist_followers=float(request.form.get("artist_followers")),
            track_duration_min=float(request.form.get("track_duration_min")),
            album_total_tracks=float(request.form.get("album_total_tracks")),
        )

        df = data.get_data_as_dataframe()

        pipeline = PredictionPipeline()
        cluster = pipeline.predict(df)

        return render_template(
            "result.html",
            cluster_id=int(cluster[0])
        )

    except Exception as e:
        return render_template("error.html", error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
