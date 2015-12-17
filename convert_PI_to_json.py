import json
import os

pi_results = os.listdir("../PI users")
output = open("PI_result.json", "w")

i = 0
for result in pi_results:
	input = open("../PI users/" + result)
	input.readline()

	text = "{"

	for line in input:
		text += line
	json_line = json.loads(text)
	json_line["id"] = result.split(".")[1]
	output.write(json.dumps(json_line))
	output.write("\n")
	print i, json_line["id"]
	i += 1
output.close()

