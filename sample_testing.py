"""
Sample code for testing new user rating, make sure you have 

1. pi_model.py
2. pi_data.py
3. pi_scaler.pkl
4. pi_model.pkl
5. pi_train_restaurant_vectors.pkl

in the file
"""

#############
import pi_model

if __name__ == "__main__":

	model = pi_model.piModel()
	model.load_model() 

	#Loading model may take a while, try to load the model just once and reuse the loaded model for testing

	# Get the top 5 recommend restaurant within the nearset 100 restaurant given a user id and his location
	# model.recommendNear100Top5(List[float] user_pi_vector, float longitude, float latitude) 
	# The selected dimension of user_pi_vector is listed in "selected_pi_feature.txt", there are totally 52 dimensions
	model.recommendNear100Top5(user_json, 100, 100)

	






