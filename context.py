from configs import config

# All the singletons, all the singletons, all the singletons...
class Context:
    _instance = None

    # for use in both single and multi
    stage = None
    model_process_map = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(Context, cls).__new__(cls)
            cls._instance._initialized = False
            cls.stage = 0
            cls.model_process_map = {}
        
        return cls._instance

context = Context()
