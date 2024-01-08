from src.generate_embeddings import (
    clean_data, train_save_word2vec, generate_embeddings_parquet)
from src.train_model import prepare_embedding, model_train
from src.predict import predict, predict_probability
from pyspark.sql import SparkSession
import pandas as pd


def pandas_to_spark(spark, path):
    loaded_data = pd.read_pickle(path)
    return spark.createDataFrame(loaded_data)


def main(retrain_model=False):
    # Initiation de la session Spark
    spark = SparkSession.builder.appName("ProjetFinal").getOrCreate()

    if retrain_model:
        clean= clean_data(spark, 'data/Data.csv')
        train_save_word2vec(clean, 'embeddings/word2vec')
        generate_embeddings_parquet(clean, 'embeddings/word2vec')

        train, test = prepare_embedding(spark, "embeddings/w2v.parquet")
        test.write.parquet("data/test.parquet")
        train.write.parquet("data/train.parquet")

        # Convertir le DataFrame Pandas en DataFrame Spark
        train = spark.read.parquet("data/train.parquet")
        model_train(train, 'models/model_lr')


    # Charger le modèle pour la prédiction
    test = spark.read.parquet("data/test.parquet")
    prediction = predict(test, 'models/model_lr')
    print(prediction)

    spark.stop()


if __name__ == "__main__":
    # Par défaut, exécute la prédiction sans réentraîner le modèle
    main()


    # Pour réentraîner le modèle et exécuter la prédiction
     #main(retrain_model=True)

