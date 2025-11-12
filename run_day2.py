import requests

API_URL = "http://localhost:8000"

def train_model():
    """
    Sends a request to the FastAPI server to train the ML model.
    """
    print("Sending request to train model at /train-model/...")
    print("This may take a moment...")
    
    try:
        response = requests.post(f"{API_URL}/train-model/")
        
        response.raise_for_status() # Raises an error for bad status codes
        
        result = response.json()
        print("\n--- Training Complete ---")
        print(f"Status: {result.get('status')}")
        print(f"Model Accuracy: {result.get('accuracy'):.2f}")
        print("Features Used:")
        for feature in result.get('features_used', []):
            print(f"- {feature}")
        print("-------------------------")
        
    except requests.exceptions.ConnectionError:
        print("\n--- ERROR ---")
        print("Could not connect to the API server.")
        print("Please ensure the FastAPI server is running with: uvicorn main:app --reload")
        print("-------------")
    except requests.exceptions.HTTPError as e:
        print(f"\n--- HTTP ERROR ---")
        print(f"Error from server: {e.response.status_code}")
        print(f"Details: {e.response.json()}")
        print("--------------------")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(e)
        print("-----------------------------------")

if __name__ == "__main__":
    train_model()