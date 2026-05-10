class DataForVisualization:
    def __init__(self, 
                 config,
                 utils,
                 **kwargs):
        """
        Args:
            config: Config instance for the algorithm
            utils: Utils instance for the algorithm
            **kwargs: Algorithm-specific data attributes
        """ 
        for var_name, var_value in vars(config).items():
            setattr(self, var_name, var_value)
        for var_name, var_value in vars(utils).items():
            setattr(self, var_name, var_value)
        for var_name, var_value in kwargs.items():
            setattr(self, var_name, var_value)
            
        # inherit algorithm_name() from config
        self.algorithm_name = config.algorithm_name
