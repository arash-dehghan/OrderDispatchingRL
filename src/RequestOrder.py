class RequestOrder(object):
	def __init__(self, current_time, destination):
		self.destination = destination
		self.origin_time = current_time
		self.deadline = -1
		self.travel_duration = -1
		self.id = -1

	def __str__(self):
		return(f'Order {self.id} ({self.destination}, {self.deadline}, {self.travel_duration})')

	def __repr__(self):
		return str(self)

	def update_state(self,current_time):
		self.state = [self.destination, self.deadline, current_time]
		self.state_str = str(self.state)

