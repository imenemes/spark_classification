from pyspark.ml.classification import RandomForestClassificationModel

def predict(embeddings, model_path):
    model = RandomForestClassificationModel.load(model_path)
    predictions = model.transform(embeddings)

    return predictions.select('prediction')
