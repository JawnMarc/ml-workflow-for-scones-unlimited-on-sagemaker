import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    boto3.client('s3').download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.prediction import Predictor


ENDPOINT = 'image-classification-2023-01-31-23-57-03-737'

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    with open(event['s3_key'], "rb") as f:
        payload = f.read()

    # Make a prediction:
    inferences = predictor.predict(payload, initial_args={'ContentType': 'application/x-image'})

    # We return the data back to the Step Function
    event["inferences"] = inferences.decode('utf-8')

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
