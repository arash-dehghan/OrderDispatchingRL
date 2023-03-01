import sys
sys.dont_write_bytecode = True
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_progress(results):
	x = list(results.requests_served.keys())
	y = np.array(list(results.requests_served.values())) / 20
	plt.xlabel('Iterations (n)')
	plt.ylabel('Average Orders Served')
	plt.title('Average Daily Orders Served Over Training')
	plt.plot(x,y)
	plt.show()

def plot_requests_throughout_day(results):
	last_index = list(results.requests_served_day_results.keys())[-1]
	original = results.requests_served_day_results[0]
	updated = results.requests_served_day_results[last_index]
	seen = results.requests_seen_day_results[0]

	print(f'The myopic version satisfied on average {results.requests_served[0]/20} orders (out of {results.requests_seen[0]/20}) and the trained after n={last_index} iterations satisfies {results.requests_served[last_index]/20} on average')
	print(f'That is an increase of {round(((results.requests_served[last_index] - results.requests_served[0]) / results.requests_seen[0]) * 100,4)}%')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Number of Requests')
	plt.title('Number of Requests Served Throughout Day')
	plt.plot(original, label = 'Myopic')
	plt.plot(updated, label = f'Trained (n={last_index} Iterations)')
	plt.plot(seen, label = 'Requests Seen')
	plt.legend()
	plt.show()

with open('Brooklyn_10_16_10.pickle', 'rb') as handle: results = pickle.load(handle)

plot_progress(results)
plot_requests_throughout_day(results)