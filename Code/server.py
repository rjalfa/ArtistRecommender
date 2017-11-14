#!/usr/bin/python

from flask import Flask, request
import json
import random
app = Flask(__name__)

mbids = [{'mbid': 'a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432', 'name': 'U2'},
{'mbid': 'bd13909f-1c29-4c27-a874-d4aaf27c5b1a', 'name': 'Fleetwood Mac'},
{'mbid': '79239441-bfd5-4981-a70c-55c3f15c1287', 'name': 'Madonna'},
{'mbid': 'ca1ca0e3-8506-40b4-ac29-dbfb60c5a0bd', 'name': 'Jars Of Clay'},
{'mbid': 'b45335d1-5219-4262-a44d-936aa36eeaed', 'name': 'Ladytron'},
{'mbid': '62dc94cc-f611-4345-87cb-b914796a4a45', 'name': 'Hooverphonic'},
{'mbid': '28a40a67-ecd3-432a-be2f-51490a7743ec', 'name': 'Helena Paparizou'},
{'mbid': '0307edfc-437c-4b48-8700-80680e66a228', 'name': 'Whitney Houston'},
{'mbid': '494e8d09-f85b-4543-892f-a5096aed1cd4', 'name': 'Mariah Carey'},
{'mbid': 'f27ec8db-af05-4f36-916e-3d57f91ecf5e', 'name': 'Michael Jackson'},
{'mbid': 'b345af35-205e-4eca-8006-029e5c20127e', 'name': 'Simple Plan'},
{'mbid': '13c10976-99f1-4cb4-8fbe-56067e91d865', 'name': 'Dolores O\'Riordan'},
{'mbid': '966e1095-b172-415c-bae5-53f8041fd050', 'name': 'David Cook'},
{'mbid': '084308bd-1654-436f-ba03-df6697104e19', 'name': 'Green Day'},
{'mbid': 'c98d40fd-f6cf-4b26-883e-eaa515ee2851', 'name': 'The Cranberries'},
{'mbid': 'a2accb58-6099-4cb5-a3c8-f6332f364db5', 'name': 'James'},
{'mbid': 'be540c02-7898-4b79-9acc-c8122c7d9e83', 'name': 'Pet Shop Boys'},
{'mbid': '455aca99-38f3-47cb-aa6b-820cf9de3c91', 'name': 'Blue Foundation'},
{'mbid': '7289d10d-d8d2-41ff-8308-c56ec9346e07', 'name': 'Texas'},
{'mbid': 'bf24ca37-25f4-4e34-9aec-460b94364cfc', 'name': 'Shakira'},
{'mbid': '77de9077-5074-4a07-9fe7-049414715ea9', 'name': 'Pixie Lott'},
{'mbid': '8538e728-ca0b-4321-b7e5-cff6565dd4c0', 'name': 'Depeche Mode'},
{'mbid': 'edbc42ea-6040-4414-8c4c-9eb7d503f64c', 'name': 'Little Boots'},
{'mbid': 'b071f9fa-14b0-4217-8e97-eb41da73f598', 'name': 'The Rolling Stones'}]

@app.route("/")
def root():
    id = request.args.get("id")
    # if random.randint(1,2) == 1:
    	# mbids[2]['name'] = 'John F. Kennedy'
    # else:
    	# mbids[2]['name'] = 'John Doe'
    obj = {"id" : id, "mbids": mbids}
    return json.dumps(obj)

@app.route("/update", methods=['POST'])
def update():
	id = request.args.get("id")
	return id

if __name__ == '__main__':
	app.run(debug=True)