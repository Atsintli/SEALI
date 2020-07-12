from decimal import Decimal
from pprint import pprint
import boto3
import requests
import sys
import os
import subprocess

def update_audio(id, analysis, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')

    table = dynamodb.Table('periferia')

    response = table.update_item(
        Key={
            'id': id
        },
        UpdateExpression="set analyzis=:a",
        ExpressionAttributeValues={
            ':a': analysis
        },
        ReturnValues="UPDATED_NEW"
    )
    return response


def scan_table(dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('periferia')

    done = False
    start_key = None
    while not done:
        if start_key:
            scan_kwargs['ExclusiveStartKey'] = start_key
        response = table.scan()
        display_audios(response.get('Items', []))
        start_key = response.get('LastEvaluatedKey', None)
        done = start_key is None

def analizador(fileName):
	print("esta analizando " + fileName)
	file_name, file_extension = os.path.splitext(fileName)
	subprocess.call(["/Users/hugosg/Desktop/essentia-extractors-v2.1_beta2/streaming_extractor_freesound", fileName, file_name + ".json"])

if __name__ == '__main__':
    #update_response = update_audio("Audio_0575fef4-2d1d-46ba-a8a9-9809d50d56db.wav", ["Larry", "Moe", "Curly"])
    #print("audio update succeeded")
    allAudios = [];

    def display_audios(audios):
        #print("newCALL")
        for audio in audios:
            allAudios.append(audio)
            #print(audio)

    scan_table()
    #print(allAudios)

    for audio in allAudios:
        print(audio['id'])
        url = 'https://periferia.s3.amazonaws.com/' + audio['id']
        r = requests.get(url, allow_redirects=True)
        open(audio['id'], 'wb').write(r.content)
        analizador(audio['id'])
