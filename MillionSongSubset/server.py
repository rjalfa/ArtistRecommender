#!/usr/bin/python

from flask import Flask, request
import json
import random
app = Flask(__name__)

mbids = [{'name': 'John Doe', 'mbid': '53b106e7-0cc6-42cc-ac95-ed8d30a3a98e'}] * 24

@app.route("/")
def root():
    id = request.args.get("id")
    if random.randint(1,2) == 1:
    	mbids[2]['name'] = 'John F. Kennedy'
    else:
    	mbids[2]['name'] = 'John Doe'
    obj = {"id" : id, "mbids": mbids}
    return json.dumps(obj)

@app.route("/update", methods=['POST'])
def update():
	id = request.args.get("id")
	return id

if __name__ == '__main__':
	app.run(debug=True)