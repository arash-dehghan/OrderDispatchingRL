import numpy as np
# np.random.seed(0)

class LearningAgent(object):
	def __init__(self, agent_id, break_length, break_start, current_time = 0):
		self.id = agent_id
		self.break_start = break_start * 60
		self.break_length = break_length * 60
		self.time_until_break = ((break_start * 60) - current_time)
		self.time_until_return = break_length * 60 if not self.time_until_break else 0
		self.orders_to_pickup = []
		self.update_state(current_time)

	def __str__(self):
		return(f'Agent {self.id} [{self.break_start}, {self.time_until_break}, {self.time_until_return}, {self.orders_to_pickup}]')

	def __repr__(self):
		return str(self)

	def update_state(self,current_time):
		self.state = [self.break_start, self.time_until_break, self.time_until_return, self.orders_to_pickup, current_time]
		self.state_str = str(self.state)