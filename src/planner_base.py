class predicate:
    """Contains a predicate, a description of a part of a current state of a system
    """
    def __init__(self, name: str, args:dict):
        """Constructor

        Args:
            name (str): name of the predicate
            args (dict): the keys are the names of the placeholders for each predicate, the values are str objects
        Example:
        
        """        
        self.name = name
        self.args = args
    def is_equal(self, other):
        """Check if two predicates are equal

        Args:
            other (predicate): the other predicate

        Returns:
            bool: True if the two predicates are equal, False otherwise
        """
        if self.name != other.name:
            return False
        if len(self.args) != len(other.args):
            return False
        for key in self.args:
            if self.args[key] != other.args[key]:
                return False
        return True
class action:
    """Contains an action, a description of an action to be performed on a system
    """
    def __init__(self, name: str, preconditions: list, effects: list):
        """Constructor

        Args:
            name (str): name of the action
            preconditions (list): list of predicates
            effects (list): list of predicates
        """        
        self.name = name
        self.preconditions = preconditions
        self.effects = effects

class state:
    """Contains a state, a description of the current state of a system
    """
    def __init__(self, predicates: list):
        """Constructor

        Args:
            predicates (list): list of predicates
        """        
        self.predicates = predicates
    def check_if_conditions_are_met(self, predicates: list):
        """Checks if the conditions are met

        Args:
            predicates (list): list of predicates
        """        
        for pred in predicates:
            if pred not in self.predicates:
                return False
            else:

        return True
    def apply_operation(self, operation):
        """Applies an operation to the state

        Args:
            operation (action): the action to be performed
        """
        
          
