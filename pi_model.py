from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import pickle
from sklearn import cross_validation
import pi_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import math
import json


# Start training 
# x_train = pickle.load(open("pi_ml_train_x.pkl"))
# y_train = pickle.load(open("pi_ml_train_y.pkl"))
# x_test = pickle.load(open("pi_ml_test_x.pkl"))
# y_test = pickle.load(open("pi_ml_test_y.pkl"))
class piModel(object):
	def __init__(self):
		self.model = None
		self.scaler = None
		self.b_vectors = None
		self.pi_features = ["Adventurousness", "Artistic interests", "Emotionality", "Imagination", "Intellect", "Authority-challenging", "Openness", "Achievement striving", "Cautiousness", "Dutifulness", "Orderliness", "Self-discipline", "Self-efficacy", "Conscientiousness", "Activity level", "Assertiveness", "Cheerfulness", "Excitement-seeking", "Outgoing", "Gregariousness", "Extraversion", "Altruism", "Cooperation", "Modesty", "Uncompromising", "Sympathy", "Trust", "Agreeableness", "Fiery", "Prone to worry", "Melancholy", "Immoderation", "Self-consciousness", "Susceptible to stress", "Emotional range", "Challenge", "Closeness", "Curiosity", "Excitement", "Harmony", "Ideal", "Liberty", "Love", "Practicality", "Self-expression", "Stability", "Structure", "Conservation", "Openness to change", "Hedonism", "Self-enhancement", "Self-transcendence"]

	def feature_selection(self, data):
		mask = [19, 40, 22, 51, 27, 11, 9, 10, 44, 14, 47, 23, 25, 49, 7, 42, 18, 35, 17, 32, 1, 31, 20, 5, 37, 43, 16, 15, 2, 12, 0, 26, 28, 50, 21, 24, 38, 46, 30, 34, 6, 48, 33, 13, 45, 8, 3, 29, 41, 36, 39, 4]
		data = np.array(data)
		return data[:, mask[:10]]

	# Validation 
	def validation(self):
		validation_set = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
		validation_result = [[] for _ in xrange(len(validation_set))]
		f1 = [[] for _ in xrange(len(validation_set))]
		precision = [[] for _ in xrange(len(validation_set))]
		recall = [[] for _ in xrange(len(validation_set))]
		parameters = []
		for i, v in enumerate(validation_set):
			print "Validation " + str(i) + " : "
			pi = pi_data.piData()
			pi.gen_validation_data_np(v[0], v[1])
			scaler = StandardScaler()
			#scale_x_train = scaler.fit_transform(self.feature_selection(pi.x_train))
			#scale_x_test = scaler.fit_transform(self.feature_selection(pi.x_test))
			scale_x_train = scaler.fit_transform(pi.x_train)
			scale_x_test = scaler.fit_transform(pi.x_test)

			# Calculate sample weight due to umbalanced labels
			weight = [len(pi.y_test)/1.0/(len(pi.y_test)-sum(pi.y_test)), len(pi.y_test)/1.0/sum(pi.y_test)]
			test_weights = [weight[j] for j in pi.y_test]
			weight = [len(pi.y_train)/1.0/(len(pi.y_train)-sum(pi.y_train)), len(pi.y_train)/1.0/sum(pi.y_train)]
			train_weights = [weight[j] for j in pi.y_train]

			""" For SVC 
			for c in [100]:#np.logspace(-2, 1, 4):
				for g in [1]:#np.logspace(0, 1, 4):
					clf = svm.SVC(class_weight='balanced', verbose=1, C=c, degree=2).fit(scale_x_train, pi.y_train)
					validation_result[i].append(clf.score(scale_x_test, pi.y_test))
					f1[i].append(f1_score(pi.y_test, clf.predict(scale_x_test)))
					precision[i].append(precision_score(pi.y_test, clf.predict(scale_x_test)))
					recall[i].append(recall_score(pi.y_test, clf.predict(scale_x_test)))
					print "Accuracy of validation " + str(i+1) + ", C = " + str(c) + ", gamma = " + str(g) + " :" + str(clf.score(scale_x_test, pi.y_test))     

					# Write the parameter to array
					if i == 0:
						parameters.append((c, g))

					print "F1 score :" + str(f1_score(pi.y_test, clf.predict(scale_x_test)))
					print "Precision :" + str(precision_score(pi.y_test, clf.predict(scale_x_test)))
					print "Recall :" + str(recall_score(pi.y_test, clf.predict(scale_x_test)))
			"""

			"""For Linear SVC 
			for c in np.logspace(-2, 2, 3):
				clf = svm.LinearSVC(class_weight='balanced', verbose=1, C=c).fit(scale_x_train, pi.y_train)
				validation_result[i].append(clf.score(scale_x_test, pi.y_test))
				f1[i].append(f1_score(pi.y_test, clf.predict(scale_x_test)))
				precision[i].append(precision_score(pi.y_test, clf.predict(scale_x_test)))
				recall[i].append(recall_score(pi.y_test, clf.predict(scale_x_test)))
				print "Accuracy of validation " + str(i+1) + ", C = " + str(c) + " :" + str(clf.score(scale_x_test, pi.y_test))     

				# Write the parameter to array
				if i == 0:
					parameters.append(c)

				print "F1 score :" + str(f1_score(pi.y_test, clf.predict(scale_x_test)))
				print "Precision :" + str(precision_score(pi.y_test, clf.predict(scale_x_test)))
				print "Recall :" + str(recall_score(pi.y_test, clf.predict(scale_x_test)))
			"""

			"""For Gradient Boosted Classifier 
			for d in [7]:#range(1, 8, 2):
				for f in [1.0]:#[1.0, 0.75, 0.5, 0.25]:
					clf = GradientBoostingClassifier(n_estimators=400, max_depth=d, verbose=1).fit(scale_x_train, pi.y_train, sample_weight=train_weights)
					validation_result[i].append(clf.score(scale_x_test, pi.y_test))
					f1[i].append(f1_score(pi.y_test, clf.predict(scale_x_test)))
					precision[i].append(precision_score(pi.y_test, clf.predict(scale_x_test)))
					recall[i].append(recall_score(pi.y_test, clf.predict(scale_x_test)))
					print "Accuracy of validation " + str(i+1) + ", Max Features = " + str(f) + ", Max Depth = " + str(d) + " :" + str(clf.score(scale_x_test, pi.y_test))     

					# Write the parameter to array
					if i == 0:
						parameters.append(d)

					print "F1 score :" + str(f1_score(pi.y_test, clf.predict(scale_x_test)))
					print "Precision :" + str(precision_score(pi.y_test, clf.predict(scale_x_test)))
					print "Recall :" + str(recall_score(pi.y_test, clf.predict(scale_x_test)))
			"""

			"""For Random Forest Classifier 
			for d in range(1, 8, 2):
				for f in [1.0]:#[1.0, 0.75, 0.5, 0.25]:
					clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_split=d, verbose=1).fit(scale_x_train, pi.y_train)
					validation_result[i].append(clf.score(scale_x_test, pi.y_test))
					f1[i].append(f1_score(pi.y_test, clf.predict(scale_x_test)))
					precision[i].append(precision_score(pi.y_test, clf.predict(scale_x_test)))
					recall[i].append(recall_score(pi.y_test, clf.predict(scale_x_test)))
					print "Accuracy of validation " + str(i+1) + ", Max Features = " + str(f) + ", Minimum Sample Split = " + str(d) + " :" + str(clf.score(scale_x_test, pi.y_test))     

					# Write the parameter to array
					if i == 0:
						parameters.append(d)

					print "F1 score :" + str(f1_score(pi.y_test, clf.predict(scale_x_test)))
					print "Precision :" + str(precision_score(pi.y_test, clf.predict(scale_x_test)))
					print "Recall :" + str(recall_score(pi.y_test, clf.predict(scale_x_test)))
			"""
			# For Logistic Regression
			for c in np.logspace(-2, 2, 3):
				clf = LogisticRegression(class_weight='balanced', penalty='l1', C=c).fit(scale_x_train, pi.y_train)
				validation_result[i].append(clf.score(scale_x_test, pi.y_test))
				f1[i].append(f1_score(pi.y_test, clf.predict(scale_x_test)))
				precision[i].append(precision_score(pi.y_test, clf.predict(scale_x_test)))
				recall[i].append(recall_score(pi.y_test, clf.predict(scale_x_test)))
				print "Accuracy of validation " + str(i+1) + ", C = " + str(c) + " :" + str(clf.score(scale_x_test, pi.y_test))     

				# Write the parameter to array
				if i == 0:
					parameters.append(c)

				print "F1 score :" + str(f1_score(pi.y_test, clf.predict(scale_x_test)))
				print "Precision :" + str(precision_score(pi.y_test, clf.predict(scale_x_test)))
				print "Recall :" + str(recall_score(pi.y_test, clf.predict(scale_x_test)))
			
			print "======================================="

		validation_mean = np.mean(np.array(validation_result), axis=0)
		f1_mean = np.mean(np.array(f1), axis=0)
		precision_mean = np.mean(np.array(precision), axis=0)
		recall_mean = np.mean(np.array(recall), axis=0)
		# One parameter
		print "Best parameter : " + " parameter1 = " + str(parameters[validation_mean.argmax()])
		# Two parameter
		#print "Best parameter : " + " parameter1 = " + str(parameters[validation_mean.argmax()][0]) + " , parameter2 = " + str(parameters[validation_mean.argmax()][1])
		print "with cross-validation accuracy: " + str(validation_mean[validation_mean.argmax()])
		print "with cross-validation f1-score: " + str(f1_mean[validation_mean.argmax()])
		print "with cross-validation precision: " + str(precision_mean[validation_mean.argmax()])
		print "with cross-validation recall: " + str(recall_mean[validation_mean.argmax()])
		
		
	def train(self):
		pi = pi_data.piData()
		pi.gen_train_data()
		scaler = StandardScaler()
		scale_x_train = scaler.fit_transform(pi.x_train)

		# Calculate sample weight due to umbalanced labels
		weight = [len(pi.y_train)/1.0/(len(pi.y_train)-sum(pi.y_train)), len(pi.y_train)/1.0/sum(pi.y_train)]
		train_weights = [weight[j] for j in pi.y_train]

		# F1 score for linear SVC
		#clf = LogisticRegression(verbose=1, C=1, penalty='l1').fit(scale_x_train, pi.y_train)
		#clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, verbose=1).fit(scale_x_train, pi.y_train, sample_weight=train_weights)
		clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_split=1).fit(scale_x_train, pi.y_train)
		print "Accuracy of train :" + str(clf.score(scale_x_train, pi.y_train))
		print "F1 score of train :" + str(f1_score(pi.y_train, clf.predict(scale_x_train)))
		print "Precision of train :" + str(precision_score(pi.y_train, clf.predict(scale_x_train)))
		print "Recall of train :" + str(recall_score(pi.y_train, clf.predict(scale_x_train)))
		print "AUC of train :" + str(roc_auc_score(pi.y_train, clf.predict(scale_x_train)))

		#### Feature Importance for linear model
		# features = [line.strip() for line in open("selected_pi_feature.txt")]
		# print "Feature importance :"
		# for i in abs(clf.coef_[0]).argsort()[::-1]:
		# 	print features[i], clf.coef_[0][i]

		features = [line.strip() + "_p" for line in open("selected_pi_feature.txt")] + [line.strip() + "_n" for line in open("selected_pi_feature.txt")]
		print "Feature importance :"
		f = []
		for i in abs(clf.feature_importances_).argsort()[::-1]:
			print features[i], clf.feature_importances_[i]	
			f.append(i)	
		print f
		pickle.dump(clf, open("pi_model.pkl", "w"))
		pickle.dump(scaler, open("pi_scaler.pkl", "w"))

	def load_model(self):
		self.model = pickle.load(open("pi_model.pkl"))
		self.scaler = pickle.load(open("pi_scaler.pkl"))
		self.b_vectors = pickle.load(open("pi_train_restaurant_vectors.pkl"))
		print "Done loading model and restaurant vectors..."

	def test(self, bussiness_id, user_vector):
		
		user_vector = user_vector*2
		if bussiness_id not in self.b_vectors:
			print "The restaurant you requested didn't rated by selected training user"
			return -1
		else:
			self.b_vector = self.b_vectors[bussiness_id]["positive"] + self.b_vectors[bussiness_id]["negative"] 

		x_test = [0]*len(self.b_vector)
		if len(user_vector) != len(self.b_vector):
			print "The user personality vector doesn't have valid dimension which is" + str(len(self.b_vector))
			return -1
		else:
			for i in xrange(len(x_test)):
				x_test[i] = user_vector[i] - self.b_vector[i]

		x_test_scale = self.scaler.transform(np.array(x_test).reshape(1,-1))
		y_predict = self.model.predict_proba(x_test_scale.reshape(1,-1))[0][1]

		return y_predict

	def recommendNear100Top5(self, user_json, longi, lati):
		user_vector = self.genPIvector(user_json)
		res_dis = []
		for b in self.b_vectors:
			dis = math.sqrt(abs(longi - self.b_vectors[b]["longitude"])**2 + abs(lati - self.b_vectors[b]["latitude"])**2)
			res_dis.append((dis, b))
		res_dis.sort()
		nearest = [f[1] for f in res_dis[:100]]
		score = [(self.test(bid, user_vector), bid) for bid in nearest]
		score.sort(reverse = True)
		return [s[1] for s in score[:5]]

	def genPIvector(self, user_json):
		a = json.loads(user_json)[80:-3])
		pi_vector = []

		for i in a["tree"]["children"][0]["children"][0]["children"]:
			# Sub personality of big five
			for j in i["children"]:
				pi_vector.append(j["percentage"])
			# Personality of big five
			pi_vector.append(i["percentage"])

		# Include needs and values to pi_vector
		for i in a["tree"]["children"][1]["children"][0]["children"]:
				pi_vector.append(i["percentage"])
		for i in a["tree"]["children"][2]["children"][0]["children"]:
				pi_vector.append(i["percentage"])
		return pi_vector

	def userPIvectorBig5(self, user_json):
		a = json.loads(user_json[80:-3])
		pi_vector = []

		for i in a["tree"]["children"][0]["children"][0]["children"]:
			# Sub personality of big five
			for j in i["children"]:
				pi_vector.append(j["percentage"])
			# Personality of big five
			pi_vector.append(i["percentage"])

		# Include needs and values to pi_vector
		for i in a["tree"]["children"][1]["children"][0]["children"]:
				pi_vector.append(i["percentage"])
		for i in a["tree"]["children"][2]["children"][0]["children"]:
				pi_vector.append(i["percentage"])
		return [pi_vector[j] for j in [6, 13, 20, 27, 34]]

	def restaurantPIvectorBig5(self, bid):
		return [self.b_vectors[bid]["positive"][i] for i in [6, 13, 20, 27, 34]]

	def getPIfeatureNameBig5(self):
		return [self.pi_features[i] for i in [6, 13, 20, 27, 34]]


if __name__ == "__main__":



	#model = piModel()

	#model.validation()
	#model.train()

	#model.load_model()
	
	#model.test("MJtnKhA3l-2ZFzhneuSccw", [0]*52)

	

	#print model.recommendNear100Top5(user_json, 100, 100)
