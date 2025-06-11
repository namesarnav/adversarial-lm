from header import * 

class Attack:
    """
    Attack: 
    All your final outputs
    
    Requires Arguments
    
    model : name the model you want to use, fine-tuned or not
    goal  : the attack recipe, see attackRecipes.md for a list of available goal functions
    constraint : define your constraint
    dataset : pass your labeled and encoded dataset
    """
    
    def __init__(self, model, goal, constraint, dataset):
        self.model = model
        self.dataset = dataset
        self.constraint = constraint
        self.goal = goal
    
    class Model:
        
        def __init__(self, model):
            self.model = model

    
    class Dataset: 
        pass
    
    