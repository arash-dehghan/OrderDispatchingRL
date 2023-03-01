from RequestOrder import RequestOrder
import numpy

class DataGenerator(object):
	def __init__(self, data, variation_percentage, speed_var, num_locations, road_speed, epoch_length, dist, df, seed):
		self.data = data
		self.variation_percentage = variation_percentage
		self.speed_var = speed_var
		self.num_locations = num_locations
		self.road_speed = road_speed
		self.epoch_length = epoch_length
		self.dist = dist
		self.df = df
		self.seed = seed
		self.np = numpy.random.RandomState(seed)

	def get_requests(self, time):
		number_of_requests = self.get_number_orders(time) + 1
		locations = self.get_locations(number_of_requests)
		return self.create_requests(time,locations)

	def get_number_orders(self, time, std=1):
		avg = self.dist[time]
		values = [v for v in range(avg-std,avg+std+1) if v >= 0]
		return self.normal_distributed_random_element(values)

	def normal_distributed_random_element(self, x):
		mean = numpy.mean(range(len(x)))
		std = numpy.std(range(len(x)))
		index = int(self.np.normal(mean, std))
		while index >= len(x):
			index = int(self.np.normal(mean, std))
		return x[int(index)]
	
	def get_locations(self, num_requests):
		# print(num_requests)
		return self.np.choice(self.df.index, size=num_requests, p=self.df.prevalence)

	def create_requests(self, time, locations):
		return [RequestOrder(time, loc) for loc in locations]

	def get_shift_start_time(self, mean = 0, std_dev = 8):
		value = self.np.normal(mean, std_dev)
		while (value < 0) or (value > 24):
			value = self.np.normal(mean, std_dev)
		return int(value)

	def create_test_scenarios(self,num_days, epoch_length):
		test_scenarios = {day: {} for day in range(num_days)}
		for day in range(num_days):
			for time in range(0,1440, epoch_length):
				test_scenarios[day][time] = self.get_requests(time)
		return test_scenarios

	# def create_locations(self, num_locations):
	# 	return [self.np.choice(self.nodes) for _ in range(num_locations)]
			





