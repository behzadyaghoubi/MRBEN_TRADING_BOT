# tools/create_dummy_model.py
import joblib


class DummyModel:
    def predict(self, X):
        try:
            import numpy as np

            return np.zeros(len(X))
        except Exception:
            return [0] * len(X)


if __name__ == "__main__":
    joblib.dump(DummyModel(), "mrben_ai_signal_filter_xgb.joblib")
    print("Dummy model saved as mrben_ai_signal_filter_xgb.joblib")
