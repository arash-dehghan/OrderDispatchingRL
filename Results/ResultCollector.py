import numpy as np

class ResultCollector(object):
	def __init__(self):
		self.requests_seen_day_results = {}
		self.requests_served_day_results = {}
		self.requests_seen = {}
		self.requests_served = {}

	def update_results(self, iteration, results):
		seen = np.sum([days_result[1] for days_result in results],0) / len(results)
		handled = np.sum([days_result[0] for days_result in results],0) / len(results)

		self.requests_seen_day_results[iteration] = seen 
		self.requests_served_day_results[iteration] = handled
		self.requests_seen[iteration] = sum(np.sum([days_result[1] for days_result in results],0))
		self.requests_served[iteration] = sum(np.sum([days_result[0] for days_result in results],0))


