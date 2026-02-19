import time

import librosa
import torch
from transformers import HubertModel, Wav2Vec2Processor
import soundfile as sf

from audio_to_flame_mapper import AudioToFlameMapper
from UI.flame_renderer import FLAME_Renderer
from models.flame_pytorch.flame_wrapper import FLAME_Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

audio, sr = librosa.load("data/audio/input/churchill_test.mp3", sr=16000)
sf.write("data/audio/input/test_audio_16khz.wav", audio, 16000)

input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

mapper = AudioToFlameMapper().to("cuda")

flame_wrapper = FLAME_Wrapper(device)
flame_renderer = FLAME_Renderer()

vertices, joints, faces = flame_wrapper.generate_mesh()
flame_renderer.start_view(vertices, joints, faces, render_joint=False)

with torch.no_grad():
    outputs = model(input_values)
    hidden_states = outputs.last_hidden_state

    flame_params = mapper(hidden_states)

    print(f"Batches: {flame_params.shape[0]}")
    print(f"Frames: {flame_params.shape[1]}")
    print(f"Params: {flame_params.shape[2]}")

    while True:
        for i in range(flame_params.shape[1]):
            start_time = time.time()
            expressions = flame_params[:, i, :50]
            jaw_pose = flame_params[:, i, 50:]

            flame_wrapper.expression = expressions
            flame_wrapper.pose[:, 3:6] = jaw_pose
            vertices, joints, faces = flame_wrapper.generate_mesh()
            flame_renderer.update_mesh_runtime(vertices, joints, faces, render_joint=False)
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.02 - elapsed))