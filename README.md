# Image Video Generator

This project is a Gradio app that generates videos from audio and images, with optional background music. It can process local files, YouTube URLs, and even generate speech and images using AI services.

## Features

- Generate videos from audio and image inputs
- Support for local audio files and YouTube URLs
- Text-to-speech generation using ElevenLabs API
- Image generation using DALL-E (OpenAI API)
- Optional background music with adjustable volume
- User-friendly Gradio interface

## Prerequisites

- Python 3.7+
- FFmpeg installed on your system
- OpenAI API key
- ElevenLabs API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/image-video-generator.git
   cd image-video-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

## Usage

Run the Gradio app:

```
python app.py
```

Then open your web browser and go to `http://localhost:7860` to access the user interface.

In the interface, you can:
- Provide audio input (file path, YouTube URL, or 'generate' for text-to-speech)
- Provide image input (file path or 'generate' to create one using DALL-E)
- Optionally add background music and adjust its volume
- Specify the output filename for the generated video

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
