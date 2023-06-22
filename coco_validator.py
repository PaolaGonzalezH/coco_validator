from datetime import datetime
import logging
from typing import List, Dict, Optional, Union
import json
from jsonschema import validate
from jsonschema import ValidationError as JsonStructureError
from pydantic import BaseModel, FilePath, ValidationError

logging.basicConfig(level=logging.INFO)


class CocoImage(BaseModel):
    id: int
    license: Optional[int]
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: datetime
    flickr_url: Optional[str]


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    area: Optional[float]
    idcrowd: Optional[int]
    segmentation: Union[List[List[float]], Dict[str, List[float]]]
    bbox: Optional[List[float]]


class CocoDataset(BaseModel):
    info: Dict[str, str]
    licences: Optional[List[Dict[str, str]]]
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[Dict[str, str]]


structure = {"type": "object",
             "properties": {
                 "info": {"type": "number"},
                 "licences": {"type": "array",
                              "items": {
                                  "type": "object",
                                  "properties": {
                                      "id": {"type": "integer"},
                                      "name": {"type": "string"}
                                  },
                                  "required": ['id', 'name']
                              }},
                 "images": {
                     "type": "array",
                     "items": {
                         "type": "object",
                         "properties": {
                             "id": {"type": "integer"},
                             "file_name": {"type": "string"},
                             "width": {"type": "integer"},
                             "height": {"type": "integer"}},
                         "required": ['id', 'file_name', 'width', 'height']}
                 },
                 "annotations": {"type": "object",
                                 "properties": {"id": {"type": "integer"},
                                                "image_id": {"type": "integer"},
                                                "category_id": {"type": "integer"},
                                                "segmentation": {"type": {"array"},
                                                                 "items": {"type": "array",
                                                                           "items": {"type": "number"}}},
                                                "area": {"type": "number"},
                                                "bbox": {"type": "array",
                                                         "items": {"type": "integer"}},
                                                "iscrowd": {"type": "integer"}},
                                 "required": ['id', 'image_id', 'category_id', 'segmentation']},
                 "categories": {"type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"}
                                    },
                                    "required": ['id', 'name']}
                                }
             }
             }


def validate_coco(path: FilePath):
    with open(path, 'r') as file:
        dataset = json.load(file)

    try:
        validate(dataset, structure)
        coco_dataset = CocoDataset.parse_obj(dataset)
        logging.info("Dataset validated.")
        return coco_dataset

    except FileNotFoundError:
        logging.error('File %s not found.', path)

    except JsonStructureError as e:
        logging.error(
            'Validation not completed due to structure error. Errors at %s', e.message)

    except ValidationError as e:
        logging.error(f'Validation failed at: %s', e.errors())

    except Exception as e:
        logging.exception('Exception raised at: %s', str(e))
