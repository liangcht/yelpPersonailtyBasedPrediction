import json
import math
import sys
import pickle

def euclideanDistance(v1, v2):

	for i in xrange(len(v1)):
		v1[i] -= v2[i]
		v1[i] *= v1[i]
	return math.sqrt(sum(v1))

input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

# Loading user personality vector (testing)
user_pi_train = open("user_pi_vector_all.json")
u_vectors = {}
for u in user_pi_train:
	u = json.loads(u)
	u_vectors[u["id"]] = u["pi_vector"]
print "Done loading user vector..."

# Loading restaurant pi vecotor
# res_pi = open("restaurant_pi_vector.json")
# b_vectors = {}
# for b in res_pi:
# 	b = json.loads(b)
# 	b_vectors[b["id"]] = {"positive":b["positive"], "negative":b["negative"]}


b_vectors = pickle.load(open("pi_train_restaurant_vectors.pkl"))
print "Done loading restaurant vector..."

# Build testing pair
test_pairs = []
i = 0
for line in input_review:
	review = json.loads(line)
	if review["user_id"] in u_vectors and review["business_id"] in b_vectors and review["stars"] in set([1, 2, 4, 5]):
		test_pairs.append((review["business_id"], review["user_id"], review["stars"] > 3))
	print "Processed " + str(i) + " reviews...\r",
	if i%10 == 0: 
		sys.stdout.flush()
	i += 1

print ""
print "Total test paris : " + str(len(test_pairs))

# Testing
FP = 0
FN = 0
TP = 0
TN = 0
i = 0
for p in test_pairs:
	b_vector = b_vectors[p[0]]
	predict_rating = (True if euclideanDistance(u_vectors[p[1]], b_vector["positive"]) <  
					euclideanDistance(u_vectors[p[1]], b_vector["negative"]) else False)

	if predict_rating:
		TP += p[2]
		FP += not p[2]
	elif not predict_rating:
		TN += not p[2]
		FN += p[2]
	print "Processed " + str(i) + " pairs...\r",
	if i%10 == 0: 
		sys.stdout.flush()
	i += 1
print ""
print "Accuracy : " + str(float(TP+TN)/len(test_pairs))
print "True Positive =" + str(TP)
print "True Negative =" + str(TN)
print "False Negative =" + str(FN)
print "False Positive =" + str(FP)

##### Testing result 12/8, only has the restaurant that have both positive and negative rating ##### 
# Done loading user vector...
# Done loading restaurant vector...
# Processed 1569263 reviews...
# Total test paris : 6142
# Processed 6141 pairs...
# Accuracy : 0.396776294367
# True Positive =1768
# True Negative =669
# False Negative =3280
# False Positive =425








