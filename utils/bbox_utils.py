# Bounding box középpontja
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center

# Bounding box szélessége
def get_bbox_width(bbox):
    bbox_width = bbox[2] - bbox[0]
    return bbox_width