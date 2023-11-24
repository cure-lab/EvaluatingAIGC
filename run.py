import argparse
from utils.logic import GenerativeDebug


# Global Variables
ATTRIBUTES = ["gender", "age", "race"]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, help="input image dir", default="")
    parser.add_argument("--out-dir", type=str, help="output dir", default="results.json")
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = get_parser()
    debug = GenerativeDebug()
    image = "./example.jpg"
    aesthetic_scores = debug.get_aesthetic_score(image)
    image_attributes = debug.get_image_attribute(image, ATTRIBUTES)
    print("aesthetic:", aesthetic_scores)
    print("attributes:", image_attributes)