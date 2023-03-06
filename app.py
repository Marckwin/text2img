# Import necessary modules
import tkinter as tk
import customtkinter as ctk 
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the tkinter app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

# Create the entry field for the prompt
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

# Create the label for displaying the generated image
lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

# Set the model ID, device, and create a stable diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

# Define a function to generate an image using the prompt
def generate(): 
    # Use autocast to automatically choose the optimal precision for the device
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    # Save the generated image and display it in the label
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

# Create a button to trigger image generation
trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

# Run the tkinter event loop
app.mainloop()
