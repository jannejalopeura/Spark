d
import random
import math
import matplotlib.pyplot as plt

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array


from pyspark.mllib.regression import LabeledPoint

#data = [
#LabeledPoint(0, [0.0]),
#LabeledPoint(1, [1.0]),
#LabeledPoint(3, [2.0]),
#LabeledPoint(2, [3.0])



# TOIMIIIII


# MUISTA SCALAAA
X= range(100)
X_max= max(X)
X = [i/float(X_max) for i in X]
#Y = [math.sqrt(i+random.random()*i) for i in range(100)]
#TOIMII Y = [i*i for i in range(4)]
Y = [i+random.random() for i in range(100)]
Y_max= max(Y)
Y = [i/float(Y_max) for i in Y]
### TOIMII

#data = [LabeledPoint(Y[i],[X[i]]) for i in range(len(X))]
#lrm = LinearRegressionWithSGD.train(sc.parallelize(data))

data=sc.parallelize(zip(X,Y))
parsedData = data.map(lambda x: LabeledPoint(x[1],[x[0]]))
lrm = LinearRegressionWithSGD.train(parsedData)

Z = [lrm.predict([i]) for i in X]

X = [i*X_max for i in X]
Y = [i*Y_max for i in Y]
Z = [i*Y_max for i in Z]


plt.plot(X,Y)
plt.plot(X,Z)
plt.show()
###
