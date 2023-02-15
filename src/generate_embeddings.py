from pyspark.sql.functions import regexp_replace, trim
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec, Word2VecModel


def train_word2vec(path):
    data = spark.read \
                .option("multiLine", True) \
                .csv(path, inferSchema=True, header=True)

    data = data.withColumn('text', regexp_replace('text', '[\s]{2,}', ''))
    data = data.withColumn('text', trim(data.text))
    data = data.withColumn('text', lower(col('text')))
    data = data.filter(data.text != '')

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(data)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    Nstopwords = remover.transform(wordsData)

    word2Vec = Word2Vec(vectorSize=150, inputCol="filtered", outputCol="features")
    w2v = word2Vec.fit(Nstopwords)

    return w2v

def save_w2v(w2v, path):
    w2v.save(path)

def generate_embeddings(path_w2v, path_data):
    w2v = Word2VecModel.load(path_w2v)

    data = spark.read \
        .option("multiLine", True) \
        .csv(path_data, inferSchema=True, header=True)

    data = data.withColumn('text', regexp_replace('text', '[\s]{2,}', ''))
    data = data.withColumn('text', trim(data.text))
    data = data.withColumn('text', lower(col('text')))
    data = data.filter(data.text != '')

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(data)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    Nstopwords = remover.transform(wordsData)

    embeddings = w2v.transform(Nstopwords)

    return embeddings