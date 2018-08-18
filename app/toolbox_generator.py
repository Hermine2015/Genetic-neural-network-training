class ToolboxGenerator:
    def register_from_configuration(self, toolbox, configuration):
        toolbox.register(configuration.name, configuration.type, configuration.lower_bound,
                         configuration.upper_bound)

    