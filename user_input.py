from source import Model
import numpy as np
import torch
from openai import OpenAI
from sklearn.decomposition import PCA


def get_embedding(client, text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None


def load_and_process_text(file_path):
    try:
        with open(file=file_path, mode="r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


def predict_text_source(embedding, model):
    try:
        with torch.no_grad():
            pred = model(embedding.unsqueeze(0))
            print(torch.sigmoid(pred))
            pred_binary = (torch.sigmoid(pred) >= 0.7).item()
            confidence = torch.sigmoid(pred).item()
            return pred_binary, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None


def user_input():
    try:
        # Initialize OpenAI client
        client = OpenAI()

        # Load the model
        try:
            model = Model()
            model.load_state_dict(torch.load("base_model_params", weights_only=True))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return

        # Get user input
        file_path = input("Import file with song: ").strip()

        # Load and process text
        text = load_and_process_text(file_path)
        if text is None:
            return

        # Get embedding
        embedding = get_embedding(client, text)
        if embedding is None:
            return

        # Process embedding - bez PCA, używamy bezpośrednio embedingu
        X = np.array(embedding)

        # Convert to tensor
        X_tensor = torch.Tensor(X)

        # Make prediction
        pred_binary, confidence = predict_text_source(X_tensor, model)
        if pred_binary is not None:
            result = "AI generated" if pred_binary == 1 else "Not AI generated"
            print(f"Result: {result} (Confidence: {confidence:.2%})")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    user_input()
