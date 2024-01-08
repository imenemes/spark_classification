from pyspark.ml.classification import LogisticRegression
import pandas as pd



# Split et préparer les vecteurs en test, train
def prepare_embedding(spark, path):
    w2v_data = spark.read.parquet(path)

    zeros = w2v_data.filter(w2v_data["Sentiment"] == 0)
    ones = w2v_data.filter(w2v_data["Sentiment"] == 1)

    train0, test0 = zeros.randomSplit([0.8, 0.2])
    train1, test1 = ones.randomSplit([0.8, 0.2])

    train = train0.union(train1).withColumnRenamed("Sentiment", "label")
    test = test0.union(test1).withColumnRenamed("Sentiment", "label")

    return train, test

# Entraîner le modèle
def model_train(train_data, saving_path):
    lr = LogisticRegression(elasticNetParam=0, regParam=0.1)
    model_lr = lr.fit(train_data)
    model_lr.save(saving_path)




