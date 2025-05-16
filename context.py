# context.py
class Context:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Context, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, stage=1):
        if not self._initialized:
            self.stage = stage
            self._initialized = True
        else:
            raise RuntimeError("Context already initialized")

    def set_stage(self, stage):
        self.stage = stage

    def get_stage(self):
        if not hasattr(self, 'stage'):
            raise KeyError("Stage not initialized in Context")
        return self.stage

# Create a single instance
context = Context()