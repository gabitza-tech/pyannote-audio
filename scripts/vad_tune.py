import argparse
import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils.hook import ProgressHook
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

    if args.model == "pyannote/segmentation-3.0":
        vad_pipeline = VoiceActivityDetection(segmentation=args.model, use_auth_token="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")
    else:
        vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",use_auth_token="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")
    
    vad_pipeline = vad_pipeline.to(torch.device("cuda"))
    
    #temp = #[x for x in range(1,19)]
    onset = [0.1]#[x/20 for x in temp]
    #temp = [x for x in range(0,19)]
    offset = [0]#[x/20 for x in temp]
    min_duration_on = [0]
    min_duration_off = [0]

    for on in onset:
        for off in offset:
            for min_on in min_duration_on:
                for min_off in min_duration_off:
                    sub_dir = f"on_{on}_off_{off}_min_on_{min_on}_min_off_{min_off}"
                    if not os.path.exists(sub_dir):
                        os.mkdir(f"{args.out_dir}/{sub_dir}")

                    if args.model == "pyannote/segmentation-3.0":
                        HYPER_PARAMETERS = {
                            "min_duration_on": min_on,
                            "min_duration_off": min_off,
                            }
                    else:
                        HYPER_PARAMETERS = {
                                "onset":on,
                                "offset":off,
                                "min_duration_on": min_on,
                                "min_duration_off": min_off,
                                }

                    print(HYPER_PARAMETERS)
                    vad_pipeline.instantiate(HYPER_PARAMETERS)

                    for file in os.listdir(args.in_dir):
                        filepath = os.path.join(args.in_dir,file)
                        print(file)
                        vad_out = vad_pipeline({"audio": filepath})
                        print("blocked?") 
                        file_id = os.path.splitext(file)[0]
                        with open(f"{args.out_dir}/{sub_dir}/{file_id}.txt","w") as f:
                            for start, end in vad_out.get_timeline():
                                dur = end-start
                                f.write(f"{start:.3f} {dur:.3f} speech\n")

