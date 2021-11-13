
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark= SparkSession.builder.appName('Football Players').getOrCreate()
dataset=spark.read.csv("football_players.csv",inferSchema=True,header=True)

columns = [ 'Weak_foot',
            'Skill_Moves',
            'Ball_Control',
            'Dribbling',
            'Marking',
            'Sliding_Tackle',
            'Standing_Tackle',
            'Aggression',
            'Reactions',
            'Attacking_Position',
            'Interceptions',
            'Vision',
            'Composure',
            'Crossing',
            'Short_Pass',
            'Long_Pass',
            'Speed',
            'Stamina',
            'Strength',
            'Balance',
            'Agility',
            'Jumping',
            'Heading',
            'Shot_Power',
            'Finishing',
            'Long_Shots',
            'Curve',
            'Freekick_Accuracy',
            'Penalties',
            'Volleys',
            'GK_Positioning',
            'GK_Diving',
            'GK_Kicking',
            'GK_Handling',
            'GK_Reflexes']

featureassembler=VectorAssembler(inputCols=columns,outputCol="Feature Column")
output=featureassembler.transform(dataset)

finalized_features = output.select("Feature Column", "Rating")

train_data,test_data=finalized_features.randomSplit([0.75,0.25])

regressor=LinearRegression(featuresCol='Feature Column', labelCol='Rating')
regressor=regressor.fit(train_data)

error = regressor.summary.rootMeanSquaredError
print(error)

pred_results=regressor.evaluate(test_data)
print(pred_results.rootMeanSquaredError)

pred_results.predictions.show(100)














