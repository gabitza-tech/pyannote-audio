import os
import sys

input_dir=sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for file in os.listdir(input_dir):
    fileid = os.path.splitext(file)[0]
    out_file = fileid+"_SPEAKER_sys.rttm"
    out_file = os.path.join(out_dir,out_file)

    filepath = os.path.join(input_dir, file)
    
    os.system(f"sed -e 's/waveform/{fileid}/g' -e 's/SPEAKER_0/S/g' {filepath} > {out_file}")
