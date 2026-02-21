import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ad_spend_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "season_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "ad_spend.csv")


class AdSpendModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Training model and creating pickle file...")
            self.model, self.encoder = self.train_and_save_model()
        else:
            print("Loading existing pickle model...")
            self.model, self.encoder = self.load_model()

    def train_and_save_model(self):
        df = pd.read_csv(DATA_PATH)

        print("Columns found:", list(df.columns))

        # rename for easier handling
        df.columns = [c.strip().lower() for c in df.columns]

        X_spend = df["ad spend ($)"]

        # encode season text
        encoder = LabelEncoder()
        X_season = encoder.fit_transform(df["season"])

        X = np.column_stack((X_spend, X_season))
        y = df["revenue ($)"]

        model = GradientBoostingRegressor()
        model.fit(X, y)

        # save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # save encoder
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)

        print("Pickle files created successfully!")
        return model, encoder

    def load_model(self):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)

        return model, encoder

    def predict(self, ad_spend, season):
        season_encoded = self.encoder.transform([season])[0]
        data = np.array([[ad_spend, season_encoded]])
        prediction = self.model.predict(data)[0]
        return round(float(prediction), 2)


if __name__ == "__main__":
    model = AdSpendModel()
    print(model.predict(10000, "Summer"))