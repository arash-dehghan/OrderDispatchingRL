from pandas import read_csv

class Environment():
	def __init__(self, num_agents, epoch_length, car_cap, data, break_length):
		self.travel_time = read_csv(f'../data/datasets/{data}/travel_time.csv', header=None).values
		self.num_locations = len(self.travel_time)
		self.num_days_trained = 0
		self.num_agents = num_agents
		self.start_epoch = 0
		self.stop_epoch = 1440
		self.epoch_length = epoch_length
		self.car_capacity = car_cap
		self.current_time = 0
		self.break_length = break_length

	def get_travel_time(self, source, destination):
		return self.travel_time[source, destination]

	def _check_break_status(self, agent):
		return False if ((self.current_time < agent.break_start) or (self.current_time > (agent.break_start + agent.break_length))) else True

	def _get_ordering_return_time(self, orders):
		ordering = [order.destination for order in orders]
		full_ordering = [0] + list(dict.fromkeys(ordering).keys()) + [0]
		location_arrival_times = {}
		time = 0
		for location_index in range(len(full_ordering)-1):
			time += self.travel_time[full_ordering[location_index]][full_ordering[location_index+1]]
			location_arrival_times[full_ordering[location_index+1]] = time
		return location_arrival_times[0]

	def simulate_vehicle_motion(self, agent):
		# If the agent is currently at the warehouse and available
		if agent.time_until_return == 0:
			# If the agent has orders it needs to deliver (i.e the action assigned to it was to deliver some new orders)
			if len(agent.orders_to_pickup) > 0:
				agent.time_until_return = max((self._get_ordering_return_time(agent.orders_to_pickup) - self.epoch_length), 0)
				agent.orders_to_pickup = []
				if agent.time_until_return < 0:
					print("NO THIS ONE")
					exit()
			else:
				agent.time_until_break -= 1
				assert len(agent.orders_to_pickup) == 0
				assert agent.time_until_return == 0
		# If the agent is not at the warehouse and is out delivering orders
		elif (agent.time_until_return > 0) and (not self._check_break_status(agent)):
			# If the agent has more orders to pickup and deliver when they get back
			if len(agent.orders_to_pickup) > 0:
				time_left_end_epoch = agent.time_until_return - self.epoch_length
				# If they won't be back at the warehouse before the end of the epoch
				if time_left_end_epoch > 0:
					agent.time_until_return = time_left_end_epoch
					if agent.time_until_return < 0:
						print("THIS ONE")
						exit()
				# If they'll reach the warehouse by the end of the epoch
				else:
					agent.time_until_return = max(self._get_ordering_return_time(agent.orders_to_pickup) - (self.epoch_length - agent.time_until_return),0)
					agent.orders_to_pickup = []
					if agent.time_until_return < 0:
						print("2 THIS ONE")
						exit()
			# If the agent doesn't have any further assigned 
			else:
				agent.time_until_return = max((agent.time_until_return - self.epoch_length), 0)
				if agent.time_until_return < 0:
						print("AGAIN THIS ONE")
						exit()
		# If the agent is not available and is on break
		elif (agent.time_until_return > 0) and (self._check_break_status(agent)):
			agent.time_until_return = max((agent.time_until_return - self.epoch_length), 0)
			if agent.time_until_return < 0:
				print("21 THIS ONE")
				exit()
			
		else:
			print(agent)
			print('Error')
			exit()





					