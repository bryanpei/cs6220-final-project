import connexion
from joblib import load

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir="swagger/")
application = app.app

# Load our pre-trained model
clf = load("./model/model-predict.joblib")


# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict("Test news headline.")
        tag("Test news headline.")
    except Exception:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


# Implement a get category function.
def categories():
    return {"Categories": list(clf.classes_)}


# Implement our predict function
def predict(headline: str):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    prediction = clf.predict([headline])

    # Return the prediction as a json
    return {"prediction": prediction[0]}


# Implement our tag function
def tag(headline: str):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    min_proba = 0.4
    proba = clf.predict_proba([headline])
    predictions = []
    for i, p in enumerate(proba):
        if p >= min_proba:
            predictions.append(clf.classes_[i])

    # Return the prediction as a json
    return {"predictions": list(predictions)}


# Read the API definition for our service from the yaml file
app.add_api("news_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    print(categories())
    app.run()
