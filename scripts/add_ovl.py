import numpy as np
import os
import sys
from pyannote.core import Segment,Annotation
from pyannote.audio.utils.signal import Binarize

def labels_to_pyannote_object(labels, uniq_name=''):
    """
    Convert the given labels to pyannote object to calculate DER and for visualization
    """
    annotation = Annotation(uri=uniq_name)
    for label in labels:
        annotation[Segment(label[0], label[1])] = label[2]
        print(label[2])
    return annotation

def get_filename_without_extension(path):
    filename_with_extension = os.path.basename(path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

rttm_path = sys.argv[1]
ovl_path = sys.argv[2]
out_file = sys.argv[3]


filename = get_filename_without_extension(rttm_path)

speaker_list = []

with open(rttm_path,"r") as f:
    lines = f.readlines()

for row in lines:
    data = row.split()
    start = float("{:.3f}".format(float(data[3])))
    end =  float("{:.3f}".format(float(data[4]))) + start
    speaker = data[7]
    speaker_list.append([start,end,speaker])

with open(ovl_path, "r") as f:
    lines = f.readlines()

ovl_list = []
for row in lines:
    data = row.split()
    start = float("{:.3f}".format(float(data[0])))
    end =  float("{:.3f}".format(float(data[1]))) + start
    speaker = data[2]
    ovl_list.append([start,end,speaker])

cat_list = speaker_list + ovl_list
# WE CREATED LIST OF SPEAKERS + OVERLAP
merged_list = sorted(cat_list, key=lambda x: x[0])

annotation_list = labels_to_pyannote_object(merged_list)

#print(annotation_list)

speaker_count = np.sum(annotation_list.discretize(resolution=0.01), axis=1, keepdims=True)

print(Binarize()(1. * (speaker_count ==2)))
results = Binarize()(1. * (speaker_count ==2)).get_timeline()

cleaned_overlap = []
for (start, end) in results:
    dur = end - start
    dur = float("{:.3f}".format(float(dur)))
    if dur > 0.01:
        cleaned_overlap.append([float("{:.3f}".format(float(start))),float("{:.3f}".format(float(end))),"overlap"])

new_list = speaker_list + cleaned_overlap
# WE CREATED LIST OF SPEAKERS + OVERLAP
final_list = sorted(new_list, key=lambda x: x[0])

#with open("intermediary.txt","a") as f:
#    for label in final_list:
#        f.write("{:.3f} {:.3f} {}\n".format(label[0],label[1],label[2]))

abs_list = []

for label in final_list:
    if label[2] != "overlap":
        abs_list.append(label)
    else:
        index = None
        
        # FIND CURRENT SPEAKER FOR OVERLAP INTERVAL
        for (i,spk_label) in enumerate(final_list):
            if label[0] >= spk_label[0] and label[1] <= spk_label[1]:
                mid_cur_spk = (spk_label[1]-spk_label[0])/2
                cur_spk = spk_label[2]
                index = i
                break 
        
        left_spk = ""
        left_spk_d = final_list[len(final_list)-1][1]
        left_spk_found = False
        right_spk = ""
        right_spk_d = final_list[len(final_list)-1][1]
        right_spk_found = False
        ovl_label = ""
        # FIND CLOSEST SPEAKER TO OVERLAP INTERVAL
        for j in range(1,20):
            if left_spk_found == True and right_spk_found == True:
                if left_spk_d <= right_spk_d:
                    ovl_label = left_spk
                else:
                    ovl_label = right_spk
                break

            if ((index-j)>=0) and (left_spk_found != True):
                if (final_list[index-j][2] != cur_spk) and (final_list[index-j][2] != "overlap"):
                    left_spk = final_list[index-j][2]
                    left_spk_found = True
                    left_spk_d = mid_cur_spk - (final_list[index-j][1]-final_list[index-j][0])/2

            if ((index+j)<=(len(final_list)-1)) and (right_spk_found != True):
                if (final_list[index+j][2] != cur_spk) and (final_list[index+j][2] != "overlap"):
                    right_spk = final_list[index+j][2]
                    right_spk_found = True
                    right_spk_d = mid_cur_spk - (final_list[index+j][1]-final_list[index+j][0])/2

        if (left_spk_found != True) or (right_spk_found != True):
            if left_spk_d <= right_spk_d:
                ovl_label = left_spk
            else:
                ovl_label = right_spk

        abs_list.append([label[0],label[1],ovl_label])

with open(out_file,"a") as f:
    for label in abs_list:
        f.write("SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n".format(filename, label[0],label[1]-label[0],label[2]))
