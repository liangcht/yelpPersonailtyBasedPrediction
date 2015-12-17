import json
import sys
import pickle

class piData(object):

	def __init__(self):
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []

	def gen_validation_data(self, _val_start, _val_end):
		input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

		#### To generate the training and testing user data ####
		user_pi_input = open("user_pi_vector_all.json").readlines()
		val_start = int(len(user_pi_input)*_val_start)
		val_end = int(len(user_pi_input)*_val_end)
		
		user_pi_train = user_pi_input[:val_start] + user_pi_input[val_end:]
		user_pi_test = user_pi_input[val_start:val_end]

		# output = open("user_pi_vecotr_train.json", "w")
		# for u in user_pi_train:
		# 	output.write(u)
		# output.close()

		# output = open("user_pi_vecotr_test.json", 'w')
		# for u in user_pi_test:
		# 	output.write(u)
		# output.close()

		# Get the positive and negative user list of each restaurant
		u_train_vectors = {}
		for u in user_pi_train:
			u = json.loads(u)
			u_train_vectors[u["id"]] = u["pi_vector"]

		b_dict = {}
		i = 0
		for line in input_review:
			review = json.loads(line)
			if not str(review["user_id"]) in u_train_vectors:
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
			print "Counted selected user of " + str(i) + " reviews ...\r", 
			if i%100 == 0: 
				sys.stdout.flush()
			i += 1
		print ""

		# Remove the restaurant in our dict that don't have enough positive visitors
		# for b in b_dict.keys():
		# 	if len(b_dict[b]["positive"]) < 5:
		# 		del b_dict[b]

		#### Get the personality vector for each restaurant
		pi_dim = len(u_train_vectors.values()[0]) # The dimemsion of personality insght vector
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
					b_vectors[b]["positive"][i] += u_train_vectors[str(u)][i]
					p_avg[i] += u_train_vectors[u][i]
				p_count += 1
			for i in xrange(pi_dim):
				b_vectors[b]["positive"][i] /= (len(b_dict[b]["positive"]) if len(b_dict[b]["positive"]) > 0 else 1)

			for u in b_dict[b]["negative"]:
				for i in xrange(pi_dim):
					b_vectors[b]["negative"][i] += u_train_vectors[str(u)][i]
					n_avg[i] += u_train_vectors[u][i]
				n_count += 1
			for i in xrange(pi_dim):
				b_vectors[b]["negative"][i] /= (len(b_dict[b]["negative"]) if len(b_dict[b]["negative"]) > 0 else 1)

			print "Calculated vectors for " + str(j) + " restaurants...\r", 
			if j%100 == 0: 
				sys.stdout.flush()
			j += 1
		print ""


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

		# output = open("restaurant_pi_vector.json", 'w')
		# i = 0
		# for b in b_vectors:
		# 	b_vectors[b]["id"] = b
		# 	# if sum(b_vectors[b]["positive"]) == 0 or sum(b_vectors[b]["negative"]) == 0:
		# 	# 	continue
		# 	output.write(json.dumps(b_vectors[b]))
		# 	output.write("\n")
		# 	print "Write " + str(i) + " restaurants...\r",
		# 	if i%100 == 0: 
		# 		sys.stdout.flush()
		# 	i += 1
		# print ""
		# output.close()


		# Loading user personality vector (testing)
		u_test_vectors = {}
		for u in user_pi_test:
			u = json.loads(u)
			u_test_vectors[u["id"]] = u["pi_vector"]


		# Build training and test data
		input_review.seek(0)
		j = 0
		for line in input_review:
			review = json.loads(line)
			if review["user_id"] in u_train_vectors and review["business_id"] in b_vectors and review["stars"] != 3:
				if sum(b_vectors[review["business_id"]]["positive"]) == 0:
					continue
				vector = [0]*len(b_vectors[review["business_id"]]["positive"])
				for i in xrange(len(vector)):
					vector[i] = b_vectors[review["business_id"]]["positive"][i] - u_train_vectors[review["user_id"]][i]
					#vector[i] = abs(b_vectors[review["business_id"]]["positive"][i] - u_train_vectors[review["user_id"]][i])
				self.x_train.append(vector)
				self.y_train.append((1 if review["stars"] > 3 else 0))
			elif review["user_id"] in u_test_vectors and review["business_id"] in b_vectors and review["stars"] != 3: 
				if sum(b_vectors[review["business_id"]]["positive"]) == 0:
					continue
				vector = [0]*len(b_vectors[review["business_id"]]["positive"])
				for i in xrange(len(vector)):
					vector[i] = b_vectors[review["business_id"]]["positive"][i] - u_test_vectors[review["user_id"]][i]
					#vector[i] = abs(b_vectors[review["business_id"]]["positive"][i] - u_test_vectors[review["user_id"]][i])
				self.x_test.append(vector)
				self.y_test.append((1 if review["stars"] > 3 else 0))
			print "Generated " + str(j) + " train/test data...\r",
			if j % 100 == 0:
				sys.stdout.flush()
			j += 1
		print ""

		print "Size of training :" + str(len(self.x_train))
		print "Size of testing :" + str(len(self.x_test))
		print "Positive in train :" + str(sum(self.y_train))
		print "Positive in test : " + str(sum(self.y_test))

	def gen_validation_data_np(self, _val_start, _val_end):
		input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

		#### To generate the training and testing user data ####
		user_pi_input = open("user_pi_vector_all.json").readlines()
		val_start = int(len(user_pi_input)*_val_start)
		val_end = int(len(user_pi_input)*_val_end)
		
		user_pi_train = user_pi_input[:val_start] + user_pi_input[val_end:]
		user_pi_test = user_pi_input[val_start:val_end]

		# output = open("user_pi_vecotr_train.json", "w")
		# for u in user_pi_train:
		# 	output.write(u)
		# output.close()

		# output = open("user_pi_vecotr_test.json", 'w')
		# for u in user_pi_test:
		# 	output.write(u)
		# output.close()

		
		# Get the positive and negative user list of each restaurant
		u_train_vectors = {}
		for u in user_pi_train:
			u = json.loads(u)
			u_train_vectors[u["id"]] = u["pi_vector"]

		b_dict = {}
		i = 0
		for line in input_review:
			review = json.loads(line)
			if not str(review["user_id"]) in u_train_vectors:
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
			print "Counted selected user of " + str(i) + " reviews ...\r", 
			if i%100 == 0: 
				sys.stdout.flush()
			i += 1
		print ""

		# Load the business profile 
		# b_cat = {}
		# for b in open("../yelp_dataset/yelp_academic_dataset_business.json"):
		# 	b = json.loads(b)
		# 	b_cat[b["business_id"]] = set(b["categories"])

		# Remove the restaurant in our dict that don't have enough positive visitors
		for b in b_dict.keys():
			if len(b_dict[b]["positive"]) < 1 or len(b_dict[b]["negative"]) < 1:
				del b_dict[b]
			
		#### Get the personality vector for each restaurant
		pi_dim = len(u_train_vectors.values()[0]) # The dimemsion of personality insght vector
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
					b_vectors[b]["positive"][i] += u_train_vectors[str(u)][i]
					p_avg[i] += u_train_vectors[u][i]
				p_count += 1
			for i in xrange(pi_dim):
				b_vectors[b]["positive"][i] /= (len(b_dict[b]["positive"]) if len(b_dict[b]["positive"]) > 0 else 1)

			for u in b_dict[b]["negative"]:
				for i in xrange(pi_dim):
					b_vectors[b]["negative"][i] += u_train_vectors[str(u)][i]
					n_avg[i] += u_train_vectors[u][i]
				n_count += 1
			for i in xrange(pi_dim):
				b_vectors[b]["negative"][i] /= (len(b_dict[b]["negative"]) if len(b_dict[b]["negative"]) > 0 else 1)

			print "Calculated vectors for " + str(j) + " restaurants...\r", 
			if j%100 == 0: 
				sys.stdout.flush()
			j += 1
		print ""


		for i in xrange(len(p_avg)):
			p_avg[i] /= p_count
			n_avg[i] /= n_count


		# Loading user personality vector (testing)
		u_test_vectors = {}
		for u in user_pi_test:
			u = json.loads(u)
			u_test_vectors[u["id"]] = u["pi_vector"]


		# Build training and test data
		input_review.seek(0)
		j = 0
		for line in input_review:
			review = json.loads(line)
			if review["user_id"] in u_train_vectors and review["business_id"] in b_vectors and review["stars"] != 3:
				if sum(b_vectors[review["business_id"]]["positive"]) == 0:
					continue
				# Positive part
				vector = [0]*len(b_vectors[review["business_id"]]["positive"])
				for i in xrange(len(vector)):
					vector[i] = b_vectors[review["business_id"]]["positive"][i] - u_train_vectors[review["user_id"]][i]
					#vector[i] = abs(b_vectors[review["business_id"]]["positive"][i] - u_train_vectors[review["user_id"]][i])
				
				# Negative part
				vector2 = [0]*len(b_vectors[review["business_id"]]["negative"])
				for i in xrange(len(vector2)):
					vector2[i] = b_vectors[review["business_id"]]["negative"][i] - u_train_vectors[review["user_id"]][i]

				self.x_train.append(vector+vector2)
				self.y_train.append((1 if review["stars"] > 3 else 0))
			elif review["user_id"] in u_test_vectors and review["business_id"] in b_vectors and review["stars"] != 3: 
				if sum(b_vectors[review["business_id"]]["positive"]) == 0:
					continue
				# Positive part
				vector = [0]*len(b_vectors[review["business_id"]]["positive"])
				for i in xrange(len(vector)):
					vector[i] = b_vectors[review["business_id"]]["positive"][i] - u_test_vectors[review["user_id"]][i]
					#vector[i] = abs(b_vectors[review["business_id"]]["positive"][i] - u_test_vectors[review["user_id"]][i])
				
				# Negative part
				vector2 = [0]*len(b_vectors[review["business_id"]]["negative"])
				for i in xrange(len(vector2)):
					vector2[i] = b_vectors[review["business_id"]]["negative"][i] - u_test_vectors[review["user_id"]][i]

				self.x_test.append(vector + vector2)
				self.y_test.append((1 if review["stars"] > 3 else 0))
			print "Generated " + str(j) + " train/test data...\r",
			if j % 100 == 0:
				sys.stdout.flush()
			j += 1
		print ""

		print "Size of training :" + str(len(self.x_train))
		print "Size of testing :" + str(len(self.x_test))
		print "Positive in train :" + str(sum(self.y_train))
		print "Positive in test : " + str(sum(self.y_test))

	def gen_train_data(self):
		input_review = open("../yelp_dataset/yelp_academic_dataset_review.json")

		user_pi_input = open("user_pi_vector_all.json").readlines()

		# Get the positive and negative user list of each restaurant
		u_train_vectors = {}
		for u in user_pi_input:
			u = json.loads(u)
			u_train_vectors[u["id"]] = u["pi_vector"]
		positive = 0
		negative = 0
		b_dict = {}
		i = 0
		for line in input_review:
			review = json.loads(line)
			if not str(review["user_id"]) in u_train_vectors:
				continue
			if review["stars"] > 3:
				positive += 1
				if review["business_id"] not in b_dict:
					b_dict[review["business_id"]] = {"positive":[review["user_id"]], "negative":[]}
				else:
					b_dict[review["business_id"]]["positive"].append(review["user_id"])
			if review["stars"] < 3:
				negative += 1
				if review["business_id"] not in b_dict:
					b_dict[review["business_id"]] = {"positive":[], "negative":[review["user_id"]]}
				else:
					b_dict[review["business_id"]]["negative"].append(review["user_id"])
			print "Counted selected user of " + str(i) + " reviews ...\r",  
			if i%100 == 0: 
				sys.stdout.flush()
			i += 1
		print ""

		print "P :", str(positive)
		print "N :", str(negative)

		# Remove the restaurant in our dict that don't have enough positive visitors
		for b in b_dict.keys():
			if len(b_dict[b]["positive"]) < 1 or len(b_dict[b]["negative"]) < 1 :
				del b_dict[b]

		#### Get the personality vector for each restaurant
		pi_dim = len(u_train_vectors.values()[0]) # The dimemsion of personality insght vector
		b_vectors = {}


		j = 0
		for b in b_dict:
			b_vectors[b] = {"positive":[0]*pi_dim, "negative":[0]*pi_dim}
			
			for u in b_dict[b]["positive"]:
				for i in xrange(pi_dim):
					b_vectors[b]["positive"][i] += u_train_vectors[str(u)][i]
			for i in xrange(pi_dim):
				b_vectors[b]["positive"][i] /= (len(b_dict[b]["positive"]) if len(b_dict[b]["positive"]) > 0 else 1)

			for u in b_dict[b]["negative"]:
				for i in xrange(pi_dim):
					b_vectors[b]["negative"][i] += u_train_vectors[str(u)][i]
			for i in xrange(pi_dim):
				b_vectors[b]["negative"][i] /= (len(b_dict[b]["negative"]) if len(b_dict[b]["negative"]) > 0 else 1)

			print "Calculated vectors for " + str(j) + " restaurants...\r", 
			if j%100 == 0: 
				sys.stdout.flush()
			j += 1
		print ""


		# Put the latitude and longitude in b_vectors
		b_file = open("../yelp_dataset/yelp_academic_dataset_business.json")
		for b in b_file:
			b = json.loads(b)
			if b["business_id"] in b_vectors:
				b_vectors[b["business_id"]]["latitude"] = b["latitude"]
				b_vectors[b["business_id"]]["longitude"] = b["longitude"]


		# Build training and test data
		input_review.seek(0)
		j = 0
		for line in input_review:
			review = json.loads(line)
			if review["user_id"] in u_train_vectors and review["business_id"] in b_vectors and review["stars"] != 3:
				if sum(b_vectors[review["business_id"]]["positive"]) == 0:
					continue
				# Positvie part
				vector = [0]*len(b_vectors[review["business_id"]]["positive"])
				for i in xrange(len(vector)):
					vector[i] = abs(b_vectors[review["business_id"]]["positive"][i] - u_train_vectors[review["user_id"]][i])

				# Negative part
				vector2 = [0]*len(b_vectors[review["business_id"]]["negative"])
				for i in xrange(len(vector2)):
					vector2[i] = abs(b_vectors[review["business_id"]]["negative"][i] - u_train_vectors[review["user_id"]][i])

				self.x_train.append(vector + vector2)
				self.y_train.append((1 if review["stars"] > 3 else 0))
			print "Generated " + str(j) + " train/test data...\r",
			if j % 100 == 0:
				sys.stdout.flush()
			j += 1
		print ""


		print "Size of train :" + str(len(self.x_train))
		print "Positive in train : " + str(sum(self.y_train))
		pickle.dump(b_vectors, open("pi_train_restaurant_vectors.pkl", "w"))
		output = open("pi_train_restaurant_list.txt", "w")
		for b in b_vectors:
			output.write(b)
			output.write("\n")
		output.close()







