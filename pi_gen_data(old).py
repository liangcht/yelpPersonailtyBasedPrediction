import json
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy
import pickle
from sklearn import cross_validation

"""
input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

# Loading user personality vector (testing)
user_pi_train = open("user_pi_vector_train.json")
u_train_vectors = {}
for u in user_pi_train:
	u = json.loads(u)
	u_train_vectors[u["id"]] = u["pi_vector"]
print "Done loading train user vector..."

# Loading user personality vector (testing)
user_pi_test = open("user_pi_vector_test.json")
u_test_vectors = {}
for u in user_pi_test:
	u = json.loads(u)
	u_test_vectors[u["id"]] = u["pi_vector"]
print "Done loading test user vector..."

# Loading restaurant pi vecotor
res_pi = open("restaurant_pi_vector.json")
b_vectors = {}
for b in res_pi:
	b = json.loads(b)
	b_vectors[b["id"]] = b["positive"]
print "Done loading restaurant vector..."

# Build training and test data
x_train = []
y_train = []
x_test = []
y_test = []

j = 0
for line in input_review:
	review = json.loads(line)
	if review["user_id"] in u_train_vectors and review["business_id"] in b_vectors and review["stars"] != 3:
		vector = [0]*len(b_vectors[review["business_id"]])
		for i in xrange(len(vector)):
			vector[i] = abs(b_vectors[review["business_id"]][i] - u_train_vectors[review["user_id"]][i])
		x_train.append(vector)
		y_train.append((1 if review["stars"] > 3 else 0))
	elif review["user_id"] in u_test_vectors and review["business_id"] in b_vectors and review["stars"] != 3:
		vector = [0]*len(b_vectors[review["business_id"]])
		for i in xrange(len(vector)):
			vector[i] = abs(b_vectors[review["business_id"]][i] - u_test_vectors[review["user_id"]][i])
		x_test.append(vector)
		y_test.append((1 if review["stars"] > 3 else 0))
	print "Processed " + str(j) + " reviews...\r",
	if j % 100 == 0:
		sys.stdout.flush()
	j += 1
print ""

pickle.dump(x_train, open("pi_ml_train_x.pkl", "w"))
pickle.dump(y_train, open("pi_ml_train_y.pkl", "w"))
pickle.dump(x_test, open("pi_ml_test_x.pkl", "w"))
pickle.dump(y_test, open("pi_ml_test_y.pkl", "w"))
"""



