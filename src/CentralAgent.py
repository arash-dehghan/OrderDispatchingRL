from collections import Counter
import itertools
import docplex.mp.model as cpx
import cplex
from copy import deepcopy
import numpy as np
import tensorflow as tf

class CentralAgent(object):
	def __init__(self, envt, num_agents, delay_allowed, car_capacity):
		self.envt = envt
		self.num_agents = num_agents
		self.delay_allowed = delay_allowed
		self.car_capacity = car_capacity

	def set_deadlines(self, order, order_id):
		order.deadline = self.envt.current_time + self.envt.travel_time[0][order.destination] + self.delay_allowed
		order.id = order_id
		order.update_state(self.envt.current_time)
		order.travel_duration = self.envt.travel_time[0][order.destination]
		return order

	def _check_break_status(self, agent):
		return False if ((self.current_time < agent.break_start) or (self.current_time > (agent.break_start + agent.break_length))) else True
	
	def _check_at_warehouse_status(self, agent):
		return True if agent.time_until_return == 0 else False
	
	def get_external_infor(self, agents):
		num_agents_on_break = sum([1 for agent in agents if self._check_break_status(agent)])
		num_agents_at_warehouse = sum([1 for agent in agents if self._check_at_warehouse_status(agent)])
		return num_agents_on_break, num_agents_at_warehouse

	def check_feasibility(self, order, agents):
		# agents = deepcopy(agents)
		returns = {agent.id : agent.time_until_return for agent in agents}
		sorted_returns = sorted(returns.items(), key=lambda x: x[1])
		matchable = False
		agent_matched_to = None
		for i,_ in sorted_returns:
			agent = agents[i]
			# (1) First check if the courier is available to be matched with orders (i.e. they're not on their break)
			if not self._check_break_status(agent):
				# (2) Check the vehicle will have room to add this order
				if (self.envt.car_capacity - (len(agent.orders_to_pickup) + 1)) >= 0:
					# Check if it can be dropped off on time (as well as other agent orders can be too)
					ordering, return_time = self._check_match(agent, order)
					if return_time != (self.envt.stop_epoch + 1):
						agent.orders_to_pickup = deepcopy(ordering)
						agent_matched_to = agent.id
						matchable = True
						break
		return matchable, agent_matched_to

	def _check_break_status(self, agent):
		return False if ((self.envt.current_time < agent.break_start) or (self.envt.current_time > (agent.break_start + agent.break_length))) else True

	def _check_match(self, agent, action):
		new_total_locs = [order.destination for order in agent.orders_to_pickup] + [action.destination]
		unique_new_locs = list(set(new_total_locs))
		orders = agent.orders_to_pickup + [action]

		best_ordering = []
		best_return_time = self.envt.stop_epoch + 1

		for ordering in itertools.permutations(unique_new_locs,len(unique_new_locs)):
			time = self.envt.current_time + agent.time_until_return
			full_ordering = [0] + list(ordering) + [0]
			location_arrival_times = {}
			for location_index in range(len(full_ordering)-1):
				time += self.envt.travel_time[full_ordering[location_index]][full_ordering[location_index+1]]
				location_arrival_times[full_ordering[location_index+1]] = time
			ordering_feasible = True
			for order in orders:
				if order.deadline < location_arrival_times[order.destination]:
					ordering_feasible = False 
					break
			###
			if ordering_feasible:
				if (agent.time_until_break < 0) or (location_arrival_times[0] < (self.envt.current_time + agent.time_until_break)):
					if (location_arrival_times[0] < best_return_time):
						best_ordering = list(ordering)
						best_return_time = location_arrival_times[0]
		final_ordering = []
		for order_loc in best_ordering:
			orders_matching_loc = [order for order in orders if order.destination == order_loc]
			final_ordering += orders_matching_loc
		return final_ordering, best_return_time

	def set_new_paths(self, agents):
		for agent in agents:
			self.envt.simulate_vehicle_motion(agent)

	def get_state(self, agents, order, num_orders, order_num, current_orders):
		states = []
		# Agent break start info
		agent_break_starts  = [agent.break_start / self.envt.stop_epoch for agent in agents]
		# Agent time until break info
		agent_time_until_breaks  = [agent.time_until_break / self.envt.stop_epoch for agent in agents]
		# Agent time until return info
		agent_time_until_returns  = [agent.time_until_return / self.envt.stop_epoch for agent in agents]
		# Agent queue sizes info
		agent_queue_sizes = [len(agent.orders_to_pickup) / self.envt.car_capacity for agent in agents]
		# Number of agents on break and number of agents at warehouse info
		num_agents_on_break, num_agents_at_warehouse = self.get_external_infor(agents)
		# Concatenation of agent information
		agent_info = agent_break_starts + agent_time_until_breaks + agent_time_until_returns + agent_queue_sizes + [num_agents_on_break / self.envt.num_agents] + [num_agents_at_warehouse / self.envt.num_agents]

		# Order duration/distance info
		order_dest_time_to_warehouse = self.envt.travel_time[0][order.destination] / self.envt.stop_epoch
		# Order origin time info
		order_origin_time = order.origin_time / self.envt.stop_epoch
		# Order deadline info
		order_deadline = order.deadline / self.envt.stop_epoch
		# Order locations of all orders at timestep
		order_locs = [-1 for _ in range(30)]
		order_locs[:len(current_orders)] = [o.destination for o in current_orders]
		# Concatenation of order information
		order_info = [order_dest_time_to_warehouse, order_origin_time, order_deadline, num_orders / 30, order_num / 30]	+ order_locs

		# State concatenation
		state = order_info + agent_info
		state = np.reshape(state, (1, -1))

		return state

	def _get_ordering_return_time(self, orders):
		ordering = [order.destination for order in orders]
		full_ordering = [0] + list(dict.fromkeys(ordering).keys()) + [0]
		location_arrival_times = {}
		time = 0
		for location_index in range(len(full_ordering)-1):
			time += self.envt.travel_time[full_ordering[location_index]][full_ordering[location_index+1]]
			location_arrival_times[full_ordering[location_index+1]] = time
		return location_arrival_times

	# def get_reward(self, agents, order, action):
	# 	if action == 0:
	# 		matchable, agent_matched_to = self.check_feasibility(order, agents)
	# 		if matchable:
	# 			drop_off_time = self._get_ordering_return_time(deepcopy(agents[agent_matched_to].orders_to_pickup))[order.destination]
	# 			reward = (order.deadline - drop_off_time)
	# 		else:
	# 			reward = 0
	# 	else:
	# 		reward = 0
	# 	return reward

	def get_reward(self, agents, order, action):
		if action == 0:
			matchable, agent_matched_to = self.check_feasibility(order, agents)
			# exit()
			if matchable:
				# drop_off_time = self._get_ordering_return_time(deepcopy(agents[agent_matched_to].orders_to_pickup))[order.destination]
				# reward = (order.deadline - drop_off_time)
				reward = 1
			else:
				reward = 0
		else:
			reward = 0
		return reward

	# def get_reward(self, agents, order, action):
	# 	if action == 0:
	# 		matchable, agent_matched_to = self.check_feasibility(order, agents)
	# 		# exit()
	# 		if matchable:
	# 			# drop_off_time = self._get_ordering_return_time(deepcopy(agents[agent_matched_to].orders_to_pickup))[order.destination]
	# 			# reward = (order.deadline - drop_off_time)
	# 			reward = 1
	# 		else:
	# 			reward = -1
	# 	else:
	# 		reward = 0
	# 	return reward




