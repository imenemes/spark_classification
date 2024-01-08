from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Word2Vec, Word2VecModel



def predict(embeddings, model_path):
    model = LogisticRegressionModel.load(model_path)
    predictions = model.transform(embeddings)
    evaluation = BinaryClassificationEvaluator()  # AUC

    return evaluation.evaluate(predictions)



def predict_probability(spark, text, word2vec_model_path, model_path):

    word2vec_data = Word2VecModel.load('models/word2vec')
    text_df = spark.createDataFrame([(text,)], ["filtered"])
    embed = word2vec_data.transform(text_df)

    # Charger le modèle de classification
    model = LogisticRegressionModel.load(model_path)

    # Prédire la probabilité d'appartenance à la classe spécifique
    predictions = model.transform(embed)
    probability = predictions.select("probability").collect()[0][0][1]

    return probability



