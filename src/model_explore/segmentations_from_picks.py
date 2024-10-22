import argparse
import copick
from tqdm import tqdm
import numpy as np
from copick_utils.segmentation import target_generator
import copick_utils.writers.write as write
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(
        description = "Generate picks segmenations from copick files."
    )
    parser.add_argument('--copick_config_path', type=str, default='copick_config_dataportal_10439.json')
    parser.add_argument('--copick_user_name', type=str, default='user0')
    parser.add_argument('--copick_segmentation_name', type=str, default='paintedPicks')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    root = copick.from_file(args.copick_config_path)

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
        write.segmentation(run, target, args.user_name, segmentationName=args.segmentation_name)

