import json
import sys

input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

#### To generate the training and testing user data ####
# user_pi_input = open("user_pi_vector.json")
# train_num = int(469*0.8)
# user_pi = user_pi_input.readlines()
# user_pi_train = user_pi[:train_num]
# user_pi_test = user_pi[train_num:]

# output = open("user_pi_vecotr_train.json", "w")
# for u in user_pi_train:
# 	output.write(u)
# output.close()

# output = open("user_pi_vecotr_test.json", 'w')
# for u in user_pi_test:
# 	output.write(u)
# output.close()


# Loading user personality vector, training
user_pi_train = open("user_pi_vector_train.json")
u_vectors = {}
for u in user_pi_train:
	u = json.loads(u)
	u_vectors[u["id"]] = u["pi_vector"]


# Get the positive and negative user list of each restaurant
b_dict = {}
i = 0
for line in input_review:
	review = json.loads(line)
	if not str(review["user_id"]) in u_vectors:
		continue
	if review["stars"] > 3:
		if review["business_id"] not in b_dict:
			b_dict[review["business_id"]] = {"positive":[review["user_id"]], "negative":[]}
		else:
			b_dict[review["business_id"]]["positive"].append(review["user_id"])
	if review["stars"] < 3:
		if review["business_id"] not in b_dict:
			b_dict[review["business_id"]] = {"positive":[], "negative":[review["user_id"]]}
		else:
			b_dict[review["business_id"]]["negative"].append(review["user_id"])
	print "Processed " + str(i) + " reviews with selected user...\r", 
	if i%100 == 0: 
		sys.stdout.flush()
	i += 1
print ""

#### Get the personality vector for each restaurant, for restaurant with missing #### 
# positive or negative user list, assign average value for positive/negative

pi_dim = len(u_vectors.values()[0]) # The dimemsion of personality insght vector
b_vectors = {}

# For counting the average
p_avg = [0]*pi_dim
p_count = 0
n_avg = [0]*pi_dim
n_count = 0

j = 0
for b in b_dict:
	b_vectors[b] = {"positive":[0]*pi_dim, "negative":[0]*pi_dim}
	
	for u in b_dict[b]["positive"]:
		for i in xrange(pi_dim):
			b_vectors[b]["positive"][i] += u_vectors[str(u)][i]
			p_avg[i] += u_vectors[u][i]
		p_count += 1
	for i in xrange(pi_dim):
		b_vectors[b]["positive"][i] /= (len(b_dict[b]["positive"]) if len(b_dict[b]["positive"]) > 0 else 1)

	for u in b_dict[b]["negative"]:
		for i in xrange(pi_dim):
			b_vectors[b]["negative"][i] += u_vectors[str(u)][i]
			n_avg[i] += u_vectors[u][i]
		n_count += 1
	for i in xrange(pi_dim):
		b_vectors[b]["negative"][i] /= (len(b_dict[b]["negative"]) if len(b_dict[b]["negative"]) > 0 else 1)

	print "Get vectors for " + str(j) + " restaurants...\r", 
	if j%100 == 0: 
		sys.stdout.flush()
	j += 1
print ""

assert(len(p_avg) == len(n_avg))
for i in xrange(len(p_avg)):
	p_avg[i] /= p_count
	n_avg[i] /= n_count

#### Assign the restaurant with missing vector as average vector ####
# mis_p = 0
# mis_n = 0
# for b in b_vectors:
# 	if sum(b_vectors[b]["positive"]) == 0:
# 		mis_p += 1
# 		b_vectors[b]["positive"] = p_avg

# 	if sum(b_vectors[b]["negative"]) == 0:
# 		mis_n += 1
# 		b_vectors[b]["negative"] = n_avg

# print "Totally " + str(len(b_dict)) + " restaurants"
# print "Totally missing " + str(mis_p) + " positive restaurant vector"
# print "Totally missing " + str(mis_n) + " negative restaurant vector"

output = open("restaurant_pi_vector.json", 'w')
i = 0
for b in b_vectors:
	b_vectors[b]["id"] = b
	# if sum(b_vectors[b]["positive"]) == 0 or sum(b_vectors[b]["negative"]) == 0:
	# 	continue
	output.write(json.dumps(b_vectors[b]))
	output.write("\n")
	print "Write " + str(i) + " restaurants...\r",
	if i%100 == 0: 
		sys.stdout.flush()
	i += 1
print ""
output.close()



