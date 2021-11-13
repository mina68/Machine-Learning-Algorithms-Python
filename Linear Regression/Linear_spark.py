
#Load the CSV file into a RDD
from pyspark import SparkContext,SparkConf
conf=SparkConf().setAppName("Football Players").setMaster("local[1]") #بدي اسم للبرنامج بتاعي و هو هيشتغل علي الجهاز بتاعي و ٢ دي معناها عدد الكورز الي عندي
SpContext=SparkContext(conf=conf) #علشان اعرف اقرا الداتاسيت بتاعتي
autoData = SpContext.textFile("football_players.csv")

autoData.cache()

#Remove the first line (contains headers)

print ("Number of elements in RDD "+ str(autoData.count()))
dataLines = autoData.filter(lambda x: 'Weak_foot' not in x)
print ("After filtering "+ str(dataLines.count()))

"""--------------------------------------------------------------------------
Cleanup Data
-------------------------------------------------------------------------"""

from pyspark.sql import Row


#Run map for cleanup
    
attList=dataLines.map(lambda l: l.split(","))
autoMap = attList.map(lambda p:  Row( Rating=float(p[0]),\
                     Weak_foot=float(p[1]), \
                     Skill_Moves=float(p[2]), \
                     Ball_Control=float(p[3]), 
                     Dribbling=float(p[4]),\
                     Marking=float(p[5]), \
                     Sliding_Tackle=float(p[6]), \
                     Standing_Tackle=float(p[7]),\
                     Aggression=float(p[8]),\
                     Reactions=float(p[9]),\
                     Attacking_Position=float(p[10]),\
                     Interceptions=float(p[11]), \
                     Vision=float(p[12]), 
                     Composure=float(p[13]),\
                     Crossing=float(p[14]), \
                     Short_Pass=float(p[15]), \
                     Long_Pass=float(p[16]),\
                     Speed=p[17],\
                     Stamina=float(p[18]),\
                     Strength=float(p[19]), \
                     Balance=float(p[20]), \
                     Agility=float(p[21]), 
                     Jumping=float(p[22]),\
                     Heading=float(p[23]), \
                     Shot_Power=float(p[24]), \
                     Finishing=float(p[25]),\
                     Long_Shots=float(p[26]),\
                     Curve=float(p[27]),\
                     Freekick_Accuracy=float(p[28]), \
                     Penalties=float(p[29]), 
                     Volleys=float(p[30]),\
                     GK_Positioning=float(p[31]), \
                     GK_Diving=float(p[32]), \
                     GK_Kicking=float(p[33]), \
                     GK_Handling=float(p[34]),\
                     GK_Reflexes=p[35]
                     
                     
                      ))
attList.cache()

#Create a Data Frame with the data. 
from pyspark.sql import SparkSession
SpSession=SparkSession(SpContext)#creating a spark session.
autoDf = SpSession.createDataFrame(attList)

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""
#See descriptive analytics.
autoDf.describe().show()


#Find correlation between predictors and target
for i in autoDf.columns:
    if not( isinstance(autoDf.select(i).take(1)[0][0], str)) :
        print( "Correlation to Rating for ", i, autoDf.stat.corr('Rating',i))


"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
SpSession=SparkSession(SpContext)
def transformToLabeledPoint(row) :
    lp = ( row["Rating"], Vectors.dense(
            [row['Weak_foot'],\
            row['Skill_Moves'],\
            row['Ball_Control'],\
            row['Dribbling'],\
            row['Marking'],\
            row['Sliding_Tackle'],\
            row['Standing_Tackle'],\
            row['Aggression'],\
           row['Reactions'],\
           row['Attacking_Position'],\
           row['Interceptions'],\
           row['Vision'],\
            row['Composure'],\
           row['Crossing'],\
            row['Short_Pass'],\
            row['Long_Pass'],\
            row['Speed'],\
            row['Stamina'],\
            row['Strength'],\
            row['Balance'],\
            row['Agility'],\
            row['Jumping'],\
            row['Heading'],\
            row['Shot_Power'],\
            row['Finishing'],\
           row[ 'Long_Shots'],\
            row['Curve'],\
            row['Freekick_Accuracy'],\
            row['Penalties'],\
            row['Volleys'],\
            row['GK_Positioning'],\
            row['GK_Diving'],\
            row['GK_Kicking'],\
           row[ 'GK_Handling'],\
            row['GK_Reflexes']
  
                        ]))
    return lp
    
autoLp = autoMap.map(transformToLabeledPoint)
autoDF = SpSession.createDataFrame(autoLp,["label", "features"])
autoDF.select("label","features").show(10)


"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""

#Split into training and testing data
(trainingData, testData) = autoDF.randomSplit([0.9, 0.1])

print ("trainingData.count() "+ str(trainingData.count()))
print ("testData.count() "+ str(testData.count()))

#Build the model on training data
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10)
lrModel = lr.fit(trainingData)

#Print the metrics
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#Predict on the test data
predictions = lrModel.transform(testData)
predictions.select("prediction","label","features").show()

#Find R2 for Linear Regression
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
evaluator.evaluate(predictions)