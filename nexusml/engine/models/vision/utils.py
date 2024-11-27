from typing import Any, Dict, List, Union

import cv2
import numpy as np

from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.exceptions import SchemaError


def detectron_preds_to_shapes(predictions: List[Dict], outputs: List[Dict], inputs: List[Dict],
                              dataframe_transforms: DataFrameTransforms) -> List[Union[List[Union[Dict, Any]], Any]]:
    """
    Function that maps predictions given by Detectron to our examples' shapes format
    Args:
        predictions (List[Dict]): predictions in Detectron's format
        outputs (List[Dict]): list of all output elements where each element is a dictionary with the element info
        inputs (List[Dict]): list of all input elements where each element is a dictionary with the element info
        dataframe_transforms (DataFrameTransforms): dataframe transforms already load that mau be needed for
                                                    creating model

    Returns:
        List[Union[List[Union[Dict, Any]], Any]] the predicted shapes for each example
    """
    shapes = []

    for pred in predictions:
        ex_pred = []
        for i, bbox in enumerate(pred["instances"]._fields["pred_boxes"].tensor):
            xmin, ymin, xmax, ymax = bbox
            if "pred_masks" in pred["instances"]._fields.keys():
                masks = pred["instances"]._fields["pred_masks"][i].cpu().numpy()
                # Find contours of the mask to convert pixels to polygon
                contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                polygon = [{
                    "x": round(point[0][0]),
                    "y": round(point[0][1])
                } for object in contours for point in object]
            else:
                polygon = [{
                    "x": round(xmin.item()),
                    "y": round(ymin.item())
                }, {
                    "x": round(xmin.item()),
                    "y": round(ymax.item())
                }, {
                    "x": round(xmax.item()),
                    "y": round(ymax.item())
                }, {
                    "x": round(xmax.item()),
                    "y": round(ymin.item())
                }]

            output_id = [o["name"] for o in outputs if o["type"] == "category"]
            input_id = [o["name"] for o in inputs if o["type"] == "image_file"]
            if len(output_id) != 1:
                raise SchemaError("Expected one single output element of type 'category'")
            if len(input_id) != 1:
                raise SchemaError("Expected one single input element of type 'image_file'")
            predicted_class = dataframe_transforms.transforms[0].categories.get_categories(
                output_id[0])[pred["instances"]._fields["pred_classes"][i].item()]
            confidence = pred["instances"]._fields["scores"][i].item()
            ex_pred.append({
                "element":
                    input_id[0],
                "polygon":
                    polygon,
                "outputs": [{
                    "element": output_id[0],
                    "value": {
                        "category": predicted_class,
                        "scores": {
                            predicted_class: confidence
                        }
                    },
                }]
            })

        shapes.append(ex_pred)

    return shapes
