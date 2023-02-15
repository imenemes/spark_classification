from pyspark.ml.classification import RandomForestClassifier



def model_train(embeddings, saving_path):
    rf = RandomForestClassifier(maxDepth=10, maxBins=10, numTrees=50)
    Modelrf = rf.fit(embeddings)
    Modelrf.save(saving_path)

