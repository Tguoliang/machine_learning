from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'AllElectronics.csv', 'r')
reader = csv.reader(allElectronicsData)
rows = [row for row in reader]
headers = rows[0]
print (rows)
print (headers)

featureList = []
labelList = []

for row in rows[1:]:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(labelList)
print(featureList)

#Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:" + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

#Vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))
print(dummyX.shape)


#Using decision tree for classfication
# clf =tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf: " + str(clf))

#visulize model
with open("allElentronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file= f)

oneRowX = dummyX[0,:]
print("oneRowx: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))


