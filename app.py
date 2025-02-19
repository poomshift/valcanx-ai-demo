import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="ValcanX AI Demo", page_icon="🎨")

# Constants
RUNPOD_ENDPOINT_ID = "gh9cabj4pp1xgs"
API_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
API_KEY = os.getenv("RUNPOD_API_KEY")
MAX_TIMEOUT = 300  # 5 minutes timeout
POLL_INTERVAL = 4  # Check status every 4 seconds
ERROR_RETRY_DELAY = 4  # Also wait 4 seconds on error before retry

def get_job_status(job_id):
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    
    try:
        response = requests.get(f"{API_BASE}/status/{job_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking job status: {str(e)}")
        return None

def generate_image(prompt, negative_prompt="bad quality, low quality, bad image, lowres", 
                  width=1024, height=1024, steps=20, guidance=3.5, seed=173805153958730,
                  lora_models=None, lora_strengths=None):
    if lora_models is None:
        lora_models = ["None"] * 4
    if lora_strengths is None:
        lora_strengths = [1.0] * 4
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Update the workflow data with all 4 LoRA models
    data = {
        "input": {
            "workflow": {
                "6": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["40", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {
                        "title": "CLIP Text Encode (Positive Prompt)"
                    }
                },
                "8": {
                    "inputs": {
                        "samples": ["31", 0],
                        "vae": ["41", 0]
                    },
                    "class_type": "VAEDecode",
                    "_meta": {
                        "title": "VAE Decode"
                    }
                },
                "9": {
                    "inputs": {
                        "filename_prefix": "ComfyUI",
                        "images": ["8", 0]
                    },
                    "class_type": "SaveImage",
                    "_meta": {
                        "title": "Save Image"
                    }
                },
                "27": {
                    "inputs": {
                        "width": width,
                        "height": height,
                        "batch_size": 1
                    },
                    "class_type": "EmptySD3LatentImage",
                    "_meta": {
                        "title": "EmptySD3LatentImage"
                    }
                },
                "31": {
                    "inputs": {
                        "seed": seed,
                        "steps": steps,
                        "cfg": 1,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1,
                        "model": ["40", 0],
                        "positive": ["38", 0],
                        "negative": ["33", 0],
                        "latent_image": ["27", 0]
                    },
                    "class_type": "KSampler",
                    "_meta": {
                        "title": "KSampler"
                    }
                },
                "33": {
                    "inputs": {
                        "text": negative_prompt,
                        "clip": ["40", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {
                        "title": "CLIP Text Encode (Negative Prompt)"
                    }
                },
                "37": {
                    "inputs": {
                        "unet_name": "flux1-dev-fp8.safetensors",
                        "weight_dtype": "default"
                    },
                    "class_type": "UNETLoader",
                    "_meta": {
                        "title": "Load Diffusion Model"
                    }
                },
                "38": {
                    "inputs": {
                        "guidance": guidance,
                        "conditioning": ["6", 0]
                    },
                    "class_type": "FluxGuidance",
                    "_meta": {
                        "title": "FluxGuidance"
                    }
                },
                "39": {
                    "inputs": {
                        "clip_name1": "clip_l.safetensors",
                        "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                        "type": "flux",
                        "device": "default"
                    },
                    "class_type": "DualCLIPLoader",
                    "_meta": {
                        "title": "DualCLIPLoader"
                    }
                },
                "40": {
                    "inputs": {
                        "lora_01": lora_models[0],
                        "strength_01": lora_strengths[0],
                        "lora_02": lora_models[1],
                        "strength_02": lora_strengths[1],
                        "lora_03": lora_models[2],
                        "strength_03": lora_strengths[2],
                        "lora_04": lora_models[3],
                        "strength_04": lora_strengths[3],
                        "model": ["37", 0],
                        "clip": ["39", 0]
                    },
                    "class_type": "Lora Loader Stack (rgthree)",
                    "_meta": {
                        "title": "Lora Loader Stack (rgthree)"
                    }
                },
                "41": {
                    "inputs": {
                        "vae_name": "ae.safetensors"
                    },
                    "class_type": "VAELoader",
                    "_meta": {
                        "title": "Load VAE"
                    }
                }
            }
        }
    }
    
    try:
        # Submit the job
        response = requests.post(f"{API_BASE}/run", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'id' not in result:
            st.error("No job ID in response")
            return None
            
        job_id = result['id']
        st.info(f"Job submitted (ID: {job_id})")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Poll for job completion
        start_time = time.time()
        last_status = None
        
        while True:
            # Check timeout
            if time.time() - start_time > MAX_TIMEOUT:
                st.error(f"Job timed out after {MAX_TIMEOUT} seconds")
                return None
            
            status_response = get_job_status(job_id)
            if status_response is None:
                time.sleep(ERROR_RETRY_DELAY)  # Use shorter error retry delay
                continue
                
            status = status_response.get('status', '')
            
            # Update progress bar based on status
            if status != last_status:
                if status == 'IN_QUEUE':
                    progress_bar.progress(0.2)
                    status_text.text("Job is queued...")
                elif status == 'IN_PROGRESS':
                    progress_bar.progress(0.6)
                    status_text.text("Generating image...")
                elif status == 'COMPLETED':
                    progress_bar.progress(1.0)
                    status_text.text("Processing completed!")
                last_status = status
            
            if status == 'COMPLETED':
                output = status_response.get('output', {})
                if isinstance(output, dict) and 'message' in output:
                    try:
                        image_data = base64.b64decode(output['message'])
                        image = Image.open(io.BytesIO(image_data))
                        return image
                    except Exception as e:
                        st.error(f"Error decoding image: {str(e)}")
                        return None
                else:
                    st.error(f"Unexpected output format: {output}")
                    return None
            elif status in ['FAILED', 'CANCELLED']:
                progress_bar.progress(1.0)
                status_text.text(f"Job {status.lower()}")
                st.error(f"Job {status.lower()}: {status_response.get('error', 'Unknown error')}")
                return None
            
            time.sleep(POLL_INTERVAL)  # Use shorter polling interval
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling the API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    # Add custom CSS for layout
    st.markdown("""
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 0rem;
            margin: auto;
        }
        .element-container img {
            width: 100%;
            height: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🎨 ValcanX AI Demo</h1>", unsafe_allow_html=True)
    
    if not API_KEY:
        st.error("Please set your RUNPOD_API_KEY in the .env file")
        return

    # Initialize LoRA variables at the start
    lora_options = ["None", "Elysia.safetensors", "p4st3l_Pastel.safetensors"]
    lora_models = []
    lora_strengths = []

    # Create main columns with adjusted ratio (1:1 instead of 3:2)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        tab1, tab2 = st.tabs(["Generate", "LoRA Models"])
        
        # LoRA tab first to initialize the variables
        with tab2:
            # Two columns for each LoRA pair
            for i in range(0, 4, 2):
                col1, col2 = st.columns(2)
                
                # First LoRA
                with col1:
                    default_index = 1 if i == 0 else 0
                    model = st.selectbox(
                        f"Model {i+1}",
                        lora_options,
                        index=default_index,
                        key=f"lora_model_{i}"
                    )
                    strength = st.slider(
                        "Strength",
                        min_value=0.0,
                        max_value=2.0,
                        value=1.0 if model != "None" else 0.0,
                        step=0.05,
                        disabled=(model == "None"),
                        key=f"lora_strength_{i}"
                    )
                    lora_models.append(model)
                    lora_strengths.append(strength)
                
                # Second LoRA
                with col2:
                    default_index = 2 if i == 0 else 0
                    model = st.selectbox(
                        f"Model {i+2}",
                        lora_options,
                        index=default_index,
                        key=f"lora_model_{i+1}"
                    )
                    # Set default strength to 0.75 for p4st3l_Pastel (second LoRA)
                    default_strength = 0.75 if (i == 0 and default_index == 2) else 1.0
                    strength = st.slider(
                        "Strength",
                        min_value=0.0,
                        max_value=2.0,
                        value=default_strength if model != "None" else 0.0,
                        step=0.05,
                        disabled=(model == "None"),
                        key=f"lora_strength_{i+1}"
                    )
                    lora_models.append(model)
                    lora_strengths.append(strength)
        
        with tab1:
            # Prompt inputs
            prompt = st.text_area(
                "Prompt",
                placeholder="Describe what you want to generate...",
                value="p4st3l photography portrait of Elysia girl in white dress, in bedroom, harsh sun light make high contrast shadow.",
                height=100
            )
            
            negative_prompt = st.text_area(
                "Negative Prompt",
                value="bad quality, low quality, bad image, lowres",
                height=50
            )
            
            # All settings in columns
            col1, col2 = st.columns(2)
            
            with col1:
                width = st.select_slider(
                    "Width",
                    options=[512, 576, 640, 704, 768, 832, 896, 960, 1024],
                    value=1024
                )
                height = st.select_slider(
                    "Height",
                    options=[512, 576, 640, 704, 768, 832, 896, 960, 1024],
                    value=1024
                )
            
            with col2:
                steps = st.slider("Steps", min_value=1, max_value=100, value=20)
                seed_value = st.number_input("Seed", value=173805153958730)
            
            # Generate button
            if st.button("Generate", use_container_width=True, type="primary"):
                with st.spinner("Creating your image..."):
                    image = generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance=3.5,  # Fixed value instead of slider
                        seed=seed_value,
                        lora_models=lora_models,
                        lora_strengths=lora_strengths
                    )
                    if image:
                        # Store the generated image in session state
                        st.session_state['generated_image'] = image

    # Right column for displaying the image
    with right_col:
        st.markdown("""
            <style>
            .placeholder-box {
                text-align: center;
                padding: 2rem;
                border: 2px dashed #ccc;
                border-radius: 8px;
                height: 512px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 37px;
            }
            </style>
            """, unsafe_allow_html=True)
            
        if 'generated_image' in st.session_state:
            st.image(st.session_state['generated_image'], use_column_width=True)
        else:
            # Placeholder when no image is generated
            st.markdown("""
                <div class="placeholder-box">
                    <p>Generated image will appear here</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 