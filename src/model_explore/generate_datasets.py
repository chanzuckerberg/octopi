"""
Generate picks segmenations from copick files.
"""

import copick
from tqdm import tqdm
import numpy as np
from copick_utils.segmentation import target_generator
import copick_utils.writers.write as write
from collections import defaultdict


copick_config_path = "copick_config_dataportal_10439.json"
root = copick.from_file(copick_config_path)

voxel_spacing = 10
target_objects = defaultdict(dict)
for object in root.pickable_objects:
    if object.is_particle:
        target_objects[object.name]['label'] = object.label
        target_objects[object.name]['radius'] = object.radius

for run in tqdm(root.runs):
    tomo = run.get_voxel_spacing(10)
    tomo = tomo.get_tomogram('wbp').numpy()
    target = np.zeros(tomo.shape, dtype=np.uint8)
    for pickable_object in root.pickable_objects:
        pick = run.get_picks(object_name=pickable_object.name, user_id="data-portal")
        if len(pick):  
            target = target_generator.from_picks(pick[0], 
                                                target, 
                                                target_objects[pickable_object.name]['radius'] * 0.8,
                                                target_objects[pickable_object.name]['label']
                                                )
    write.segmentation(run, target, "user0", segmentationName='paintedPicks')

