import json

input = open("PI_result.json")
output = open("user_pi_vector_all.json", "w")
output2 = open("selected_pi_feature.txt", "w")

for line in input:
	user = {}
	a = json.loads(line)
	user["id"] = a["id"]
	user["pi_vector"] = []
	for i in a["tree"]["children"][0]["children"][0]["children"]:
		# Sub personality of big five
		for j in i["children"]:
			user["pi_vector"].append(j["percentage"])
		# Personality of big five
		user["pi_vector"].append(i["percentage"])

	# Include needs and values to pi_vector
	for i in a["tree"]["children"][1]["children"][0]["children"]:
			user["pi_vector"].append(i["percentage"])
	for i in a["tree"]["children"][2]["children"][0]["children"]:
			user["pi_vector"].append(i["percentage"])

	output.write(json.dumps(user))
	output.write("\n")
output.close()

input.seek(0)

line = input.readline()

a = json.loads(line)
for i in a["tree"]["children"][0]["children"][0]["children"]:
	# Sub personality of big five
	for j in i["children"]:
		output2.write(j["name"])
		output2.write("\n")
	# Personality of big five
	output2.write(i["name"])
	output2.write("\n")

# Include needs and values to pi_vector
for i in a["tree"]["children"][1]["children"][0]["children"]:
	output2.write(i["name"])
	output2.write("\n")
for i in a["tree"]["children"][2]["children"][0]["children"]:
	output2.write(i["name"])
	output2.write("\n")

output2.close()