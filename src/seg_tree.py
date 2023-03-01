import numpy as np


class SegTree():
    """ Base Segment Tree Class with binary heap implementation that push
    values as a Queue(FIFO).
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cycle = 0
        self.size = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.experiences = np.zeros(capacity)  # basically leaves of the tree

    def push(self, value: float):
        """ Push a value into the tree by calling the update method. Push
        function overrides values when the tree is full  """
        self.update(self._cycle, value)
        self._cycle = (self._cycle + 1) % self.capacity
        if self.size < self.capacity: # size cannot be bigger than capacity
            self.size += 1

    def update(self, index, value):
        raise NotImplementedError


class SumTree(SegTree):
    """ A Binary tree with the property that a parent node is the sum of its
    two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)

    def get(self, value: float) -> int:
        """ Return the index (ranging from 0 to max capcaity) that corresponds
        to the given value """
        if value > self.tree[0]:  # Root is the sum of the all leaves in the tree
            raise ValueError("Value is greater than the root")

        root = 0
        while True:
            left_index = 2 * root + 1
            right_index = 2 * root + 2
            if left_index >= len(self.tree) - 1:
                break
            if value <= self.tree[left_index]:
                root = left_index
            else:
                value = value - self.tree[left_index]
                root = right_index
        return root - self.capacity + 1

    def update(self, index: int, value: float):
        """ Update the value of the given index (ranging from 0 to max
        capacity) with the given value.
        Arguments:
            - index: It is basically the buffer index and experiences
            are the leaves. So index should be converted to the tree index.
        """
        assert value >= 0, "Value cannot be negative"

        tree_cycle = index + self.capacity - 1
        self.experiences[index] = value
        self.tree[tree_cycle] = value
        while tree_cycle != 0:
            tree_cycle = (tree_cycle - 1) // 2
            self.tree[tree_cycle] = self.tree[2 * tree_cycle + 1] + self.tree[2 * tree_cycle + 2]

