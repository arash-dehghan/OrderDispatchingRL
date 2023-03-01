class Experience(object):
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = state
        self.action = action
        self.reward = reward 
        self.next_state = next_state
        self.terminal = terminal

