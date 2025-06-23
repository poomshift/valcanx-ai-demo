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
st.set_page_config(page_title="Prateep AI Demo", page_icon="🎨")

# Constants
RUNPOD_ENDPOINT_ID = "gh9cabj4pp1xgs"
API_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
API_KEY = os.getenv("RUNPOD_API_KEY")
MAX_TIMEOUT = 300  # 5 minutes timeout
POLL_INTERVAL = 4  # Check status every 4 seconds
ERROR_RETRY_DELAY = 4  # Also wait 4 seconds on error before retry

# LoRA configuration
LORA_CONFIG = {
    "None": {
        "preview": None,
        "description": "No LoRA model applied",
        "placeholder_color": "#f0f0f0",
        "triggerword": ""
    },
    # "Elysia.safetensors": {
    #     "preview": "assets/lora_previews/elysia_preview.jpg",
    #     "description": "Elysia character style - perfect for anime-style portraits",
    #     "placeholder_color": "#ffd6e0",
    #     "triggerword": "Elysia"
    # },
    # "p4st3l_Pastel.safetensors": {
    #     "preview": "assets/lora_previews/pastel_preview.jpg",
    #     "description": "Soft pastel style - creates dreamy, ethereal images",
    #     "placeholder_color": "#e6f3ff",
    #     "triggerword": "p4st3l"
    # },
    # "nolan_style_flux_v2.safetensors": {
    #     "preview": "assets/lora_previews/nolan_preview.jpg",
    #     "description": "Christopher Nolan inspired style - dramatic lighting and cinematic look",
    #     "placeholder_color": "#2c2c2c",
    #     "triggerword": "nolan style"
    # },
    "Benjarong_flux_v1.safetensors": {
        "preview": "assets/lora_previews/benjarong_preview.jpg",
        "description": "Benjarong traditional Thai ceramic style - ornate patterns and gold details",
        "placeholder_color": "#ffd700",
        "triggerword": "Benjarong ornate designs"
    },
    "WaiKru_flux_v1.safetensors": {
        "preview": "assets/lora_previews/waikru_preview.jpg",
        "description": "WaiKru traditional Thai school ceremony",
        "placeholder_color": "#ff9933",
        "triggerword": "WaiKru, traditional Thai school ceremony"
    },
    "ChickenGreenCurry.safetensors": {
        "preview": "assets/lora_previews/chickencurry_preview.jpg",
        "description": "Thai Green Curry style - vibrant green colors and food photography",
        "placeholder_color": "#228b22",
        "triggerword": "ChickenGreenCurry,a yellow-colored curry or stew"
    },
    "SomTumThai.safetensors": {
        "preview": "assets/lora_previews/somtum_preview.jpg",
        "description": "Som Tum Thai papaya salad style - fresh ingredients and Thai cuisine",
        "placeholder_color": "#ff6347",
        "triggerword": "SomTumThai"
    },
    "Pratukhong_BombV1_flux_v1.safetensors": {
        "preview": "assets/lora_previews/pratukhong_preview.jpg",
        "description": "Pratukhong, A photograph of an ancient temple, likely a Buddhist temple, features a prominent, ornate archway with intricate carvings and a central figure, likely Buddha, seated in a meditative pose",
        "placeholder_color": "#ff4500",
        "triggerword": "Pratukhong, A photograph of an ancient temple, likely a Buddhist temple, features a prominent, ornate archway with intricate carvings and a central figure, likely Buddha, seated in a meditative pose"
    }
}

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
        lora_models = ["None"] * 2
    if lora_strengths is None:
        lora_strengths = [1.0] * 2
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Update the workflow data with 2 LoRA models
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
                        "lora_03": "None",
                        "strength_03": 0.0,
                        "lora_04": "None",
                        "strength_04": 0.0,
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
        .lora-preview {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .lora-card {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: #f8f9fa;
        }
        .lora-description {
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
        }
        .lora-placeholder {
            width: 100%;
            height: 120px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 0.9em;
        }
        .lora-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🎨 Prateep AI Demo</h1>", unsafe_allow_html=True)
    
    if not API_KEY:
        st.error("Please set your RUNPOD_API_KEY in the .env file")
        return

    # Initialize session state for tracking LoRA selections
    if 'previous_loras' not in st.session_state:
        st.session_state.previous_loras = ["None", "None"]
    
    if 'base_prompt' not in st.session_state:
        st.session_state.base_prompt = "product photography of Benjarong ornate designs bowl on wooden table in cozy thai kitchen, vegetables on background"

    # Initialize LoRA variables at the start
    lora_options = list(LORA_CONFIG.keys())
    lora_models = []
    lora_strengths = []

    # Create main columns with adjusted ratio (1:1 instead of 3:2)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Prompt inputs (moved to top)
        st.markdown("### Prompt")
        
        # Initialize prompt value
        prompt_value = st.session_state.get('updated_prompt', st.session_state.base_prompt)
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe what you want to generate...",
            value=prompt_value,
            height=100,
            key="prompt_input"
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt",
            value="bad quality, low quality, bad image, lowres",
            height=50
        )
        
        # LoRA Selection Section
        st.markdown("### LoRA Models")
        
        # Create a grid of LoRA selections
        col1, col2 = st.columns(2)
        
        # First LoRA
        with col1:
            st.markdown("#### LoRA 1")
            default_index = 1 if len(lora_options) > 1 else 0
            model1 = st.selectbox(
                "Model",
                lora_options,
                index=default_index,
                key="lora_model_0"
            )
            
            # Show preview or placeholder
            if LORA_CONFIG[model1]["preview"] and os.path.exists(LORA_CONFIG[model1]["preview"]):
                st.image(LORA_CONFIG[model1]["preview"], use_column_width=True)
            else:
                placeholder_color = LORA_CONFIG[model1]["placeholder_color"]
                st.markdown(f"""
                    <div class="lora-placeholder" style="background-color: {placeholder_color}">
                        {LORA_CONFIG[model1]["description"]}
                    </div>
                """, unsafe_allow_html=True)
            
            # Show trigger word if available
            if LORA_CONFIG[model1]["triggerword"]:
                st.info(f"Trigger word: **{LORA_CONFIG[model1]['triggerword']}**")
            
            strength1 = st.slider(
                "Strength",
                min_value=0.0,
                max_value=2.0,
                value=1.0 if model1 != "None" else 0.0,
                step=0.05,
                disabled=(model1 == "None"),
                key="lora_strength_0"
            )
            lora_models.append(model1)
            lora_strengths.append(strength1)
        
        # Second LoRA
        with col2:
            st.markdown("#### LoRA 2")
            default_index = 0 if len(lora_options) > 2 else 0
            model2 = st.selectbox(
                "Model",
                lora_options,
                index=default_index,
                key="lora_model_1"
            )
            
            # Show preview or placeholder
            if LORA_CONFIG[model2]["preview"] and os.path.exists(LORA_CONFIG[model2]["preview"]):
                st.image(LORA_CONFIG[model2]["preview"], use_column_width=True)
            else:
                placeholder_color = LORA_CONFIG[model2]["placeholder_color"]
                st.markdown(f"""
                    <div class="lora-placeholder" style="background-color: {placeholder_color}">
                        {LORA_CONFIG[model2]["description"]}
                    </div>
                """, unsafe_allow_html=True)
            
            # Show trigger word if available
            if LORA_CONFIG[model2]["triggerword"]:
                st.info(f"Trigger word: **{LORA_CONFIG[model2]['triggerword']}**")
            
            strength2 = st.slider(
                "Strength",
                min_value=0.0,
                max_value=2.0,
                value=1.0 if model2 != "None" else 0.0,
                step=0.05,
                disabled=(model2 == "None"),
                key="lora_strength_1"
            )
            lora_models.append(model2)
            lora_strengths.append(strength2)

        # Check if LoRA selection changed and update prompt
        current_loras = [model1, model2]
        if current_loras != st.session_state.previous_loras:
            # Get current prompt text
            current_prompt = prompt if 'prompt' in locals() else st.session_state.get('updated_prompt', st.session_state.base_prompt)
            
            # Update the prompt with trigger words
            trigger_words = []
            for model in current_loras:
                if model != "None" and LORA_CONFIG[model]["triggerword"]:
                    trigger_word = LORA_CONFIG[model]["triggerword"]
                    # Only add trigger word if it's not already in the prompt
                    if trigger_word.lower() not in current_prompt.lower():
                        trigger_words.append(trigger_word)
            
            # Build new prompt
            if trigger_words:
                # Add new trigger words to the beginning of existing prompt
                new_prompt = " ".join(trigger_words) + " " + current_prompt
            else:
                new_prompt = current_prompt
            
            # Update session state
            st.session_state.previous_loras = current_loras
            
            # Force rerun to update the text area
            if 'updated_prompt' not in st.session_state or st.session_state.updated_prompt != new_prompt:
                st.session_state.updated_prompt = new_prompt
                st.rerun()

        # Update base prompt when user manually edits (remove trigger words to get base)
        current_trigger_words = []
        for model in current_loras:
            if model != "None" and LORA_CONFIG[model]["triggerword"]:
                current_trigger_words.append(LORA_CONFIG[model]["triggerword"])
        
        # Extract base prompt by removing trigger words from the beginning
        if current_trigger_words:
            # Create a copy of the prompt to work with
            temp_prompt = prompt
            # Remove each trigger word if it's at the beginning
            for trigger_word in current_trigger_words:
                if temp_prompt.lower().startswith(trigger_word.lower() + " "):
                    temp_prompt = temp_prompt[len(trigger_word) + 1:]
                elif temp_prompt.lower().startswith(trigger_word.lower()):
                    temp_prompt = temp_prompt[len(trigger_word):]
            st.session_state.base_prompt = temp_prompt.strip()
        else:
            st.session_state.base_prompt = prompt
        
        # Generation settings
        st.markdown("### Generation Settings")
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