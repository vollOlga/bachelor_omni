class DummyGripper:
    def __init__(self, prim_path=None):
        self.closed = False
        self.holding_object = False

    def set_translation(self, x, y, z):
        pass

    def set_rotation(self, w, x, y, z):
        pass

    def close_gripper(self):
        self.closed = True

    def open_gripper(self):
        self.closed = False
        self.holding_object = False

    def request_compute(self):
        pass

    def is_gripper_closed(self):
        return self.closed

    def hold_object(self):
        if self.closed:
            self.holding_object = True

    def release_object(self):
        self.holding_object = False

    def is_holding_object(self):
        return self.holding_object
