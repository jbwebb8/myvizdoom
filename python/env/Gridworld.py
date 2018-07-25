from env.Wrapper import Wrapper

class Gridworld(Wrapper):
    """
    Wrapper class for Gridworld environment to confer DoomGame functions
    needed by Agent subclasses.
    """

    def __init__(self, env):

        # Initialize base class
        Wrapper.__init__(self)

        self.env = env
        self.num_channels = 3

    def get_available_buttons(self):
        """Returns list of one-hot vectors encoding actions"""

        actions = []
        for i in range(self.env.actions):
            a = [0] * self.env.actions
            a[i] = 1
            actions.append(a)
        return actions

    def get_available_buttons_size(self):
        """Returns number of actions"""

        return self.env.actions

    def get_screen_channels(self):
        """Returns number of channels in screen"""

        return self.num_channels

    def get_available_game_variables(self):
        
        return []