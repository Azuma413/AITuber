import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# Load LLM for text generation
llm = pipeline("text-generation", model="gpt-2")

# Load Stable Diffusion for image generation
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion = stable_diffusion.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_story(summary, num_pages):
    # Generate story text based on the summary and number of pages
    prompt = f"Generate a paper puppet show-style story with {num_pages} pages based on this summary: {summary}"
    story_text = llm(prompt, max_length=1024, num_return_sequences=1)[0]['generated_text']
    
    # Generate images for each page of the story
    images = []
    for i in range(num_pages):
        image_prompt = f"Create an image for page {i+1} of the paper puppet show story"
        image = stable_diffusion(image_prompt)["sample"][0]
        images.append(image)
    
    return story_text, images

if __name__ == "__main__":
    summary = input("Enter the summary of the story: ")
    num_pages = int(input("Enter the number of pages: "))
    story_text, images = generate_story(summary, num_pages)
    
    print("Generated Story Text:")
    print(story_text)
    
    # Save images
    for i, image in enumerate(images):
        image.save(f"page_{i+1}.png")
