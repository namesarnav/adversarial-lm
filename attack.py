from header import * 


class AttackRecipe(abc.ABC):
    """Abstract base class for attack recipes."""
    
    @abstractmethod
    def attack(self, input_text: str, label: int, model_wrapper: 'ModelWrapper') -> Dict[str, Any]:
        """Perform an attack on a single input.
        
        Args:
            input_text: Input text to attack.
            label: Original label of the input.
            model_wrapper: Wrapped model for predictions.
            
        Returns:
            Dictionary containing attack results (e.g., perturbed text, success).
        """
        pass

class TextFoolerRecipe(AttackRecipe):
    """Implementation of TextFooler attack recipe."""
    
    def attack(self, input_text: str, label: int, model_wrapper: 'ModelWrapper') -> Dict[str, Any]:
        # Placeholder for TextFooler logic
        # In a real implementation, you'd use synonym replacement, semantic similarity checks, etc.
        perturbed_text = input_text.replace("good", "great")  # Simplified example
        success = model_wrapper.predict(perturbed_text) != label
        return {
            "original_text": input_text,
            "perturbed_text": perturbed_text,
            "original_label": label,
            "success": success
        }

class TextBuggerRecipe(AttackRecipe):
    """Implementation of TextBugger attack recipe."""
    
    def attack(self, input_text: str, label: int, model_wrapper: 'ModelWrapper') -> Dict[str, Any]:
        # Placeholder for TextBugger logic
        # In a real implementation, you'd use character-level perturbations, word splits, etc.
        perturbed_text = input_text + "!!!"  # Simplified example
        success = model_wrapper.predict(perturbed_text) != label
        return {
            "original_text": input_text,
            "perturbed_text": perturbed_text,
            "original_label": label,
            "success": success
        }

class ModelWrapper:
    """Wrapper for machine learning models to standardize prediction interface."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, text: str) -> int:
        """Predict the label for a given text.
        
        Args:
            text: Input text to classify.
            
        Returns:
            Predicted label (integer).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=-1).item()
