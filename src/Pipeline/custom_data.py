import pandas as pd
from src.exception import CustomException
from src.logger.logger import logging


class CustomData:
    def __init__(
        self,
        track_name,
        artist_name,
        artist_genres,
        album_name,
        track_popularity,
        artist_popularity,
        artist_followers,
        track_duration_min,
        album_total_tracks,
    ):
        self.track_name = track_name
        self.artist_name = artist_name
        self.artist_genres = artist_genres
        self.album_name = album_name
        self.track_popularity = track_popularity
        self.artist_popularity = artist_popularity
        self.artist_followers = artist_followers
        self.track_duration_min = track_duration_min
        self.album_total_tracks = album_total_tracks

    def get_data_as_dataframe(self):
        try:
            logging.info("Creating input dataframe")

            return pd.DataFrame(
                {
                    "track_name": [self.track_name],
                    "artist_name": [self.artist_name],
                    "artist_genres": [self.artist_genres],
                    "album_name": [self.album_name],
                    "track_popularity": [self.track_popularity],
                    "artist_popularity": [self.artist_popularity],
                    "artist_followers": [self.artist_followers],
                    "track_duration_min": [self.track_duration_min],
                    "album_total_tracks": [self.album_total_tracks],
                }
            )

        except Exception as e:
            raise CustomException(e)
