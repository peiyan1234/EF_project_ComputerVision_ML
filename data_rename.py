import json
import os
import glob
import sys

number = 0
dir_whereamI = os.getcwd()
path, local_directory = os.path.split(dir_whereamI)

try:
    json_filenames = sorted(glob.glob('*.json'))
    if len(json_filenames) == 0:
        print('No json file exist.')
    else:
        for json_filename in json_filenames:

            original_image = json_filename.replace(".json",".png")
            enhanced_image = json_filename.replace(".json","_enhanced.png")
            ROI_mask_image = json_filename.replace(".json","_ROI_mask.png")

            try:
                os.replace(json_filename, local_directory + '_' + str(number) + ".json")
            except:
                print(json_filename + 'do not exist.')
                continue

            try:
                os.replace(original_image, local_directory + '_' + str(number) + ".png")
    
            except:
                print(original_image + 'do not exist.')
                continue
    
            try:
                os.replace(enhanced_image, local_directory + '_' + str(number) + "_enhanced.png")

            except:
                print(enhanced_image + 'do not exist.')
                continue
    
            try:
                os.replace(ROI_mask_image, local_directory + '_' + str(number) + "_ROI_mask.png")

            except:
                print(ROI_mask_image + 'do not exist.')
                continue

            number += 1


except:
    print("Unexpected error:", sys.exc_info()[0])


