from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    """
    A class to generate text using Hugging Face transformers.

    This class wraps the Hugging Face text generation pipeline to provide
    an easy interface for generating text from a model like GPT-2.  The user can specify
    parameters like the model name, the length of the text that is generated, and the
    number of sequences to return.
    """
    
    def __init__(self, model_name="gpt2", max_length=100, num_return_sequences=1):
        """
        Initialize the TextGenerator class with the specified model and configuration.

        Args:
            model_name (str):           The name of the pre-trained Hugging Face model (default is gpt2).
            max_length (int):           The maximum length of the generated text (default is 100 tokens).
            num_return_sequences (int): The number of generated sequences to return (default is 1).

        Initializes the tokenizer, model, and generator pipeline using the provided model_name.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt):
        """
        Generates text based on the provided input.

        Args:
            prompt (str): THe input text to begin text generation.

        Returns:
            list: A list of generated text sequences.

        Generates text based on the prompt, using the specified maximum length and
        number of return sequences.  The function returns a list of generated texts.
        """
        outputs = self.generator(prompt, max_length=self.max_length, num_return_sequences=self.num_return_sequences)
        return [output["generated_text"] for output in outputs]

# Initialize Flask App
app = Flask(__name__)

# Create and instantiate a TextGenerator object with model gpt2
text_generator = TextGenerator(model_name="gpt2", max_length=100, num_return_sequences=1)

@app.route("/")
def home():
    """
    Renders the homepage of the web app.

    This route serves the frontend HTML page, where users can input prompts for text generation.

    Returns:
        str: The rendered HTML page (index.html).
    """
    return render_template("index.html")  

@app.route("/generate", methods=["POST"])
def generate_text():
    """
    API endpoint to generate text from a user-provided prompt.

    This route accepts a POST request with a JSON body containing a prompt.  It uses the
    TextGenerator class to generate text based on the prompt and returns the generated
    text as a JSON response.

    Args:
        None
    
    Returns:
        Response: A JSON response containing the generated text or an error message.
            - Success: {"generated_texts}: ["Generated text"]}
            - Error: {"error": "No prompt provided"}

    Raises:
        400: If no prompt is provided in the request body.
    """
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    generated_texts = text_generator.generate(prompt)
    return jsonify({"generated_texts": generated_texts})

if __name__ == "__main__":
    """
    Starts the Flask application in debug mode.

    This block of code runs the Flask app when the script is executed directly.
    It starts the Flask development server on "localhost:5000" in debug mode, which
    allows for automatic reloading and detailed error messages.

    Args:
        None

    Returns:
        None 
    """
    app.run(debug=True)
