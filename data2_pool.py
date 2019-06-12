import json
import os
import glob
import sys
import shutil

dir_training_pool = '/home/alvinli/Desktop/EF/dataset/EF-training-Pool'

dir_A2C_patients = '/home/alvinli/Desktop/EF/dataset/A2C_patients'
dir_A4C_patients = '/home/alvinli/Desktop/EF/dataset/A4C_patients'

subdirectories_A2C = sorted([dir for dir in os.listdir(dir_A2C_patients) if os.path.isdir(os.path.join(dir_A2C_patients, dir))])
subdirectories_A4C = sorted([dir for dir in os.listdir(dir_A4C_patients) if os.path.isdir(os.path.join(dir_A4C_patients, dir))])

data_dict = {}
data_dict['A2C'] = {}
data_dict['A4C'] = {}

if len(subdirectories_A2C) != 0:
    try:        
        number = 0
        for dir in subdirectories_A2C:
            json_filenames = sorted(glob.glob(os.path.join(dir_A2C_patients,dir,'*.json')))
            if len(json_filenames) == 0:
                print('No json file exist.')
            else:
                for json_filename in json_filenames:
                    original_image = json_filename.replace(".json",".png")
                    ROI_mask_image = json_filename.replace(".json","_ROI_mask.png")

                    new_imagename = os.path.join(dir_training_pool, 'A2C_' + str(number) + '.png')
                    new_maskname = os.path.join(dir_training_pool, 'A2C_' + str(number) + '_ROI_mask.png')

                    shutil.copy(original_image, new_imagename)
                    shutil.copy(ROI_mask_image, new_maskname)

                    data_dict['A2C'][new_imagename] = new_maskname

                    number += 1

            
    except:
        print("Unexpected error:", sys.exc_info()[0])
    
if len(subdirectories_A4C) != 0:
    try:
        number = 0
        for dir in subdirectories_A4C:
            json_filenames = sorted(glob.glob(os.path.join(dir_A4C_patients,dir,'*.json')))
            if len(json_filenames) == 0:
                print('No json file exist.')
            else:
                for json_filename in json_filenames:
                    original_image = json_filename.replace(".json",".png")
                    ROI_mask_image = json_filename.replace(".json","_ROI_mask.png")

                    new_imagename = os.path.join(dir_training_pool, 'A4C_' + str(number) + '.png')
                    new_maskname = os.path.join(dir_training_pool, 'A4C_' + str(number) + '_ROI_mask.png')

                    shutil.copy(original_image, new_imagename)
                    shutil.copy(ROI_mask_image, new_maskname)

                    data_dict['A4C'][new_imagename] = new_maskname

                    number += 1
    except:
        print("Unexpected error:", sys.exc_info()[0])

with open(os.path.join(dir_training_pool,'datasheet.json'), 'w') as json_file:
    json.dump(data_dict, json_file)
