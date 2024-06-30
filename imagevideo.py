import gradio as gr
import os
import sys
import subprocess
import time
import re
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
from yt_dlp import YoutubeDL

MAX_FILENAME_LENGTH = 100
ELEVENLABS_VOICE_ID = "WWr4C8ld745zI3BiA8n7"
DEFAULT_BG_MUSIC_VOLUME = 0.2

# Custom functions from the original script

def sanitize_filename(filename):
    sanitized = re.sub(r'[^\w\-_ ]', '-', filename)
    sanitized = re.sub(r'-+', '-', sanitized)
    sanitized = sanitized.strip('-')
    return sanitized

def generate_image_prompt(description, is_retry=False):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_content = "You are a helpful assistant that creates high-quality image prompts for DALL-E based on user descriptions."
    if len(description) < 15:
        system_content += " Always include visual elements that represent music or audio in your prompts, even if not explicitly mentioned in the description."
    if is_retry:
        system_content += " The previous prompt violated content policy. Please create a new prompt that avoids potentially sensitive or controversial topics."
    
    user_content = f"Create a detailed, high-quality image prompt for DALL-E based on this description: {description}"
    if len(description) < 15:
        user_content += " Ensure to include visual elements representing music or audio."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

def generate_image(prompt, audio_filename, max_retries=3):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    for attempt in range(max_retries):
        print(f"Generating image (Attempt {attempt + 1}/{max_retries})")
        
        try:
            response = client.images.generate(
                prompt=prompt,
                model="dall-e-3",
                n=1,
                quality="hd",
                size="1024x1024"
            )
            
            image_url = response.data[0].url
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content))
            
            audio_name = os.path.splitext(os.path.basename(audio_filename))[0]
            img_path = f"{audio_name}_image.png"
            img.save(img_path)
            print(f"Image saved: {img_path}")
            
            return img_path

        except Exception as e:
            if "content_policy_violation" in str(e):
                print(f"Content policy violation. Regenerating prompt...")
                prompt = generate_image_prompt(prompt, is_retry=True)
            else:
                print(f"Error generating image: {e}")
            
            if attempt == max_retries - 1:
                print("Max retries reached. Image generation failed.")
                return None
    
    return None

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': {'default': '%(title)s.%(ext)s'},
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info['title']
        description = info.get('description', '')
        
        sanitized_title = sanitize_filename(title)
        
        if len(sanitized_title) > MAX_FILENAME_LENGTH:
            sanitized_title = sanitized_title[:MAX_FILENAME_LENGTH]
        
        filename = f"{sanitized_title}.%(ext)s"
        ydl_opts['outtmpl']['default'] = filename
        ydl.download([url])
    
    output_filename = f"{sanitized_title}.wav"
    print(f"Audio downloaded: {output_filename}")
    return output_filename, sanitized_title, description

def generate_speech(text, voice_id=None):
    print("Generating speech with ElevenLabs...")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key is not set.")
    
    if not voice_id:
        voice_id = ELEVENLABS_VOICE_ID
    
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(tts_url, headers=headers, json=data, stream=True)
    if response.ok:
        title = generate_title_from_text(text)
        audio_filename = f"{title}.mp3"
        with open(audio_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Audio generated: {audio_filename}")
        return audio_filename, title, text
    else:
        print(response.text)
        raise Exception("Failed to generate speech with ElevenLabs.")

def generate_title_from_text(text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates concise and descriptive titles for audio files based on their text content."},
            {"role": "user", "content": f"Generate a concise and descriptive title for an audio file based on this text: {text}"}
        ]
    )
    title = response.choices[0].message.content.strip()
    title = re.sub(r'[^\w\s-]', '', title)
    title = re.sub(r'\s+', '_', title)
    return title[:MAX_FILENAME_LENGTH]

def infer_image_description(title, description=None):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    is_short = len(title.split()) <= 3

    if description:
        prompt = f"Based on the title '{title}' and description '{description}', describe an image that would be suitable for a video thumbnail or cover art for this audio content. The description should be detailed and visually rich."
    else:
        prompt = f"Based on the title '{title}', describe an image that would be suitable for a video thumbnail or cover art for this audio content. The description should be detailed and visually rich."

    if is_short:
        prompt += f" Since the title is short, make sure to include visual elements that represent audio or music in your description, even if not directly mentioned in the title."
    
    system_content = "You are a creative assistant that generates detailed image descriptions based on titles and descriptions for audio content."
    if is_short:
        system_content += " For short titles, always include visual elements that represent music or audio in your descriptions."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_background_music(bg_music_source):
    if os.path.isfile(bg_music_source):
        return bg_music_source
    elif "youtube.com" in bg_music_source or "youtu.be" in bg_music_source:
        print("Downloading background music from YouTube...")
        return download_youtube_audio(bg_music_source)[0]  # Return only the file path
    else:
        print("Invalid background music input. Please provide a valid file path or YouTube URL.")
        return None

def generate_video(image_path, audio_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size

    resolution = f"{width}x{height}"
    video_bitrate = "5M"
    audio_bitrate = "320k"
    
    ffmpeg_command = [
        "ffmpeg",
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-vf", f"scale={resolution}",
        "-b:v", video_bitrate,
        "-y", output_path
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
        return False

def generate_video_with_background(main_audio_path, image_path, bg_music_path, output_path, bg_music_volume):
    with Image.open(image_path) as img:
        width, height = img.size

    resolution = f"{width}x{height}"
    video_bitrate = "5M"
    audio_bitrate = "320k"

    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        main_audio_path
    ]
    try:
        result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
        main_audio_duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting audio duration: {e}")
        return False

    total_duration = main_audio_duration + 2
    
    ffmpeg_command = [
        "ffmpeg",
        "-loop", "1",
        "-i", image_path,
        "-i", main_audio_path,
        "-i", bg_music_path,
        "-filter_complex", 
        f"[1:a]aformat=fltp:44100:stereo,adelay=500|500,apad=pad_dur=2[a1];"
        f"[2:a]aformat=fltp:44100:stereo,volume={bg_music_volume},aloop=loop=-1:size=2e+09,"
        f"afade=t=out:st={main_audio_duration}:d=2[a2];"
        f"[a1][a2]amix=inputs=2:duration=first[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-pix_fmt", "yuv420p",
        "-t", str(total_duration),
        "-vf", f"scale={resolution}",
        "-b:v", video_bitrate,
        "-y", output_path
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video generated: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
        return False

# Gradio app function
def generate_video_gradio(audio_input, image_input, bg_music_input, bg_music_volume, output_filename):
    # Process audio input
    if audio_input.startswith("http"):
        audio_path, title, description = download_youtube_audio(audio_input)
    elif audio_input == "generate":
        text = "This is a sample text for speech generation."  # You may want to add an input for this
        audio_path, title, description = generate_speech(text)
    else:
        audio_path = audio_input
        title = os.path.splitext(os.path.basename(audio_path))[0]
        description = None

    # Process image input
    if image_input == "generate":
        image_description = infer_image_description(title, description)
        image_prompt = generate_image_prompt(image_description)
        image_path = generate_image(image_prompt, audio_path)
    else:
        image_path = image_input

    # Process background music
    if bg_music_input:
        bg_music_path = get_background_music(bg_music_input)
    else:
        bg_music_path = None

    # Generate video
    if not output_filename.endswith('.mp4'):
        output_filename += '.mp4'

    if bg_music_path:
        success = generate_video_with_background(audio_path, image_path, bg_music_path, output_filename, bg_music_volume)
    else:
        success = generate_video(image_path, audio_path, output_filename)

    if success:
        return output_filename
    else:
        return "Video generation failed."

# Create Gradio interface
iface = gr.Interface(
    fn=generate_video_gradio,
    inputs=[
        gr.Textbox(label="Audio Input (file path, YouTube URL, or 'generate')"),
        gr.Textbox(label="Image Input (file path or 'generate')"),
        gr.Textbox(label="Background Music (file path or YouTube URL, optional)"),
        gr.Slider(minimum=0, maximum=1, step=0.1, default=0.2, label="Background Music Volume"),
        gr.Textbox(label="Output Filename")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Image Video Generator",
    description="Generate a video from audio and image, with optional background music."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
