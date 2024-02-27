from pyannote.audio import Pipeline
import os
import sys
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

input_dir = sys.argv[1]
out_dir = sys.argv[2]
# run the pipeline on an audio file
pipeline = pipeline.to(torch.device("cuda"))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# dump the diarization output to disk using RTTM format
for file in os.listdir(input_dir):
    filepath = os.path.join(input_dir, file)
    print(filepath)
  
    waveform, sample_rate = torchaudio.load(filepath)

    with ProgressHook() as hook:
        diarization = pipeline({"waveform":waveform,"sample_rate":sample_rate}, hook=hook)

    out_file = os.path.join(out_dir,os.path.splitext(file)[0]+".rttm")
    print(out_file)

    with open(out_file, "w") as rttm:
        diarization.write_rttm(rttm)
        
