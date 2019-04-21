import os
import numpy as np

def read_data_into_mem(raw=True, result=True, meta=False, segments=False):
	data = {}

	if raw:
		for dirname, dirnames, filenames in os.walk('./chiron/raw'):
			for file in filenames:
				name = file.split('.')[0]
				
				if name not in data:
					data[name] = {}

				with open('./chiron/raw/{}'.format(file)) as f:
					data[name]['raw'] = {
						"data": f.readlines()[0]
					}

	if result:
		for dirname, dirnames, filenames in os.walk('./chiron/result'):
			for file in filenames:
				name = file.split('.')[0]
				
				if name not in data:
					data[name] = {}

				with open('./chiron/result/{}'.format(file)) as f:
					lines = f.readlines()
					data[name]['result'] = {
						"name": lines[0],
						"sequence": lines[1],
						"quality": lines[3],
					}

	if meta:
		for dirname, dirnames, filenames in os.walk('./chiron/meta'):
			pass

	if segments:
		for dirname, dirnames, filenames in os.walk('./chiron/segments'):
			pass