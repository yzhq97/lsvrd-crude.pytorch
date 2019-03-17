import sys, os
sys.path.insert(0, os.getcwd())
import json
import numpy as np

meta_path = "data/gqa/objects/gqa_objects_info.json"
meta = json.load(open(meta_path))

widths = []
heights = []
areas = []

for image_id, entry in meta.items():
    widths.append(entry["width"])
    heights.append(entry["height"])
    areas.append(entry["width"] * entry["height"])

widths = np.array(widths)
heights = np.array(heights)
areas = np.array(areas)
areas = areas ** 0.5

print("width: mean %.2f min %d max %d" % (widths.mean(), widths.min(), widths.max()))
print("height: mean %.2f min %d max %d" % (heights.mean(), heights.min(), heights.max()))
print("sqrt(area): mean %.2f min %.2f max %.2f" % (areas.mean(), areas.min(), areas.max()))