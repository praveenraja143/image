import os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get the token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the client
client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)

# Generate the image
image = client.text_to_image(
    prompt="Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

# ✅ Save the image to a desired folder
save_path = os.path.join(os.getcwd(), "generated_image.png")  # current folder
# OR: save_path = r"C:\Users\YourName\Pictures\generated_image.png"

image.save(save_path)
print(f"✅ Image saved to: {save_path}")
