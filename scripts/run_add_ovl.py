import os
import sys

in_dir = sys.argv[1]
ovl_dir = sys.argv[2]
out_dir = sys.argv[3]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for file in os.listdir(in_dir):
    path = os.path.join(in_dir,file)
    out_file = os.path.join(out_dir,file.split(".")[0]+".rttm")
    ovl_path = os.path.join(ovl_dir, file.split(".")[0]+".txt")

    os.system(f"python3 scripts/add_ovl.py {path} {ovl_path} {out_file}")
