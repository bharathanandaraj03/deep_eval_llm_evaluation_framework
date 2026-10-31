import io

import numpy as np
import soundfile as sf

def add_noise(wav: bytes, snr_db:float,seed: int =  0) -> bytes:
    audio,sr = sf.read(io.BytesIO(wav), dtype="float32")
    noise_power = np.mean(audio**2) / 10 ** (snr_db/10)
    noise = np.random.default_rng(seed).normal(0,np.sqrt(noise_power),audio.shape)
    buf = io.BytesIO()
    sf.write(buf,np.clip(audio+noise,-1,1),sr , format="WAV")
    return buf.getvalue()