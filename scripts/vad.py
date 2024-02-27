import argparse
import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection

from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Model

import torch

def get_args():
    parser = argparse.ArgumentParser(
        description="Run Pyannote speech activity detection."
    )
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        help="Path to the input directory containing the wav files.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path to the output directory where the label file will be written.",
        )
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Path to the model. If not provided, we use the pretrained model from HuggingFace.",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    model = Model.from_pretrained(args.model, use_auth_token="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")
    ovl_pipeline = VoiceActivityDetection(segmentation=model)
    
    ovl_pipeline = ovl_pipeline.to(torch.device("cpu"))#(torch.device("cuda"))
   
    print(model.specifications.powerset)
    min_duration_on = [0]
    min_duration_off = [0]

    for min_on in min_duration_on:
        for min_off in min_duration_off:
            sub_dir = f"min_on_{min_on}_min_off_{min_off}"
            if not os.path.exists(os.path.join(args.out_dir,sub_dir)):
                os.mkdir(f"{args.out_dir}/{sub_dir}")

            HYPER_PARAMETERS = {
                            "min_duration_on": min_on,
                            "min_duration_off": min_off,
                    }
            if args.model == "pyannote/segmentation":
                    HYPER_PARAMETERS = {
                            "min_duration_on": min_on,
                            "min_duration_off": min_off,
                            "onset": 0.1,
                            "offset": 0,
                            }
            print(HYPER_PARAMETERS)
            ovl_pipeline.instantiate(HYPER_PARAMETERS)

            for file in os.listdir(args.in_dir):
                filepath = os.path.join(args.in_dir,file)
                ovl_out = ovl_pipeline({"audio": filepath})
                file_id = os.path.splitext(file)[0]
                print(file_id)
                with open(f"{args.out_dir}/{sub_dir}/{file_id}.txt","w") as f:
                    for start, end in ovl_out.get_timeline():
                        dur = end-start
                        f.write(f"{start:.3f} {dur:.3f} speech\n")

