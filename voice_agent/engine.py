
from .schemas import TextToAudioRequest

import asyncio
import os
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
from pydub import AudioSegment

from .schemas import TextToAudioRequest, TextToAudioResponse

# Define paths
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "outputs")
ambience_dir = os.path.join(base_dir, "assets", "ambience")
os.makedirs(output_dir, exist_ok=True)

# Load the model and processor
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Define available voices for different languages and accents
voice_presets = {
    "en-us": "v2/en_speaker_6",
    "en-gb": "v2/en_speaker_3",
    "es-es": "v2/es_speaker_1",
    "fr-fr": "v2/fr_speaker_2",
    "de-de": "v2/de_speaker_4",
    "it-it": "v2/it_speaker_5",
    "pt-br": "v2/pt_speaker_7",
    "hi-in": "v2/hi_speaker_0",
    "ja-jp": "v2/ja_speaker_8",
    "zh-cn": "v2/zh_speaker_9",
}


async def generate_audio_logic(request: TextToAudioRequest) -> TextToAudioResponse:
    """
    Core logic for generating audio from text using suno/bark model.
    """
    print(f"Generating audio for: '{request.text}' in {request.language} with a {request.accent} accent.")

    voice_preset = voice_presets.get(f"{request.language}-{request.accent}", "v2/en_speaker_6")  # Default to en-us

    inputs = processor(request.text, voice_preset=voice_preset, return_tensors="pt")

    # Generate audio
    audio_array = model.generate(**inputs, do_sample=True, fine_temperature=request.creativity, coarse_temperature=request.stability)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Save the audio file
    sample_rate = model.generation_config.sample_rate
    audio_file_name = f"generated_audio_{request.language}_{request.accent}.wav"
    audio_file_path = os.path.join(output_dir, audio_file_name)
    wavfile.write(audio_file_path, sample_rate, audio_array)

    # If ambience is requested, mix it in
    if request.ambience != "none":
        try:
            # Load generated speech
            speech = AudioSegment.from_wav(audio_file_path)

            # Load ambience track
            ambience_file = os.path.join(ambience_dir, f"{request.ambience}.wav")
            if not os.path.exists(ambience_file):
                print(f"Warning: Ambience file not found: {ambience_file}. Skipping ambience.")
            else:
                ambience = AudioSegment.from_wav(ambience_file)

                # Mix speech with ambience (overlay)
                # Adjust ambience volume to be in the background
                mixed = speech.overlay(ambience - 10)  # Reduce ambience volume by 10 dB

                # Export mixed audio
                mixed_file_name = f"generated_audio_with_{request.ambience}.wav"
                mixed_file_path = os.path.join(output_dir, mixed_file_name)
                mixed.export(mixed_file_path, format="wav")
                audio_file_path = mixed_file_path

        except Exception as e:
            print(f"Error during audio mixing: {e}. Returning original audio.")

    return TextToAudioResponse(audio_file=audio_file_path, message="Audio generated successfully.")


async def test_generate_audio_logic():
    print("Testing audio generation logic without ambience...")
    test_request_no_ambience = TextToAudioRequest(
        text="Hello, this is a test of the voice agent logic without any background sound.",
        language="en",
        accent="us",
        ambience="none"
    )
    response = await generate_audio_logic(test_request_no_ambience)
    print(f"Response: {response}")

    print("\nTesting audio generation logic with 'cafe' ambience...")
    # Note: This test requires a 'cafe.wav' file in the assets/ambience directory
    test_request_with_ambience = TextToAudioRequest(
        text="I'll have a black coffee, please.",
        language="en",
        accent="us",
        ambience="cafe"
    )
    response = await generate_audio_logic(test_request_with_ambience)
    print(f"Response: {response}")

    print("\nTesting audio generation logic with custom creativity and stability...")
    test_request_custom = TextToAudioRequest(
        text="This is a test with a more creative and less stable voice.",
        language="en",
        accent="us",
        ambience="none",
        creativity=0.8,
        stability=0.4
    )
    response = await generate_audio_logic(test_request_custom)
    print(f"Response: {response}")


if __name__ == '__main__':
    # Example of how to test this module directly
    asyncio.run(test_generate_audio_logic())
