from pyspark.sql.functions import regexp_replace, trim
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf




# fonction qui permet de traiter une classification binaire
def ratings(rating):
    if rating>3 and rating<=5:
        return 1
    if rating>0 and rating<=3:
        return 0

# nettoyage des données
def clean_data(spark, path_data):
    data = spark.read \
        .option("multiLine", True) \
        .csv(path_data, inferSchema=True, header=True)
    data = data.na.drop()
    data = data.withColumn("Sentiment", udf(ratings, IntegerType())(data["Rating"]))

    data = data.withColumn('Review', regexp_replace('Review', '[\s]{2,}', ''))
    data = data.withColumn('Review', trim(data.Review))
    data = data.withColumn('Review', lower(col('Review')))
    data = data.filter(data.Review != '')

    tokenizer = Tokenizer(inputCol="Review", outputCol="words")
    tokenized = tokenizer.transform(data)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cleaned = remover.transform(tokenized)


    return cleaned

# enregistrer les embeddings
def train_save_word2vec(data, path):
    word2vec = Word2Vec(vectorSize=150, inputCol="filtered", outputCol="features")
    w2v = word2vec.fit(data)
    w2v.save(path)

# générer et stocker les données vectoriser et les préparer

def generate_embeddings_parquet(data, path):
    word2vec_data = Word2VecModel.load(path)
    # Appliquer le modèle entraîné à nos données
    w2v_data = word2vec_data.transform(data)

    # Sauvegarder le DataFrame Spark au format Parquet
    w2v_data.write.parquet("models/w2v.parquet")



