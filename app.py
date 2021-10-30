import os
from flask import Flask
from flask_restful import Api
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
# import requests
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import boto3
import json

def instantiate_reps_from_s3(bucket_name, fn):
    # load from s3
    s3_resource = boto3.resource('s3')
    print('Downloading from S3')
    s3_resource.Object(bucket_name, fn).download_file(f'intents.json')
    
    # load from local
    intents = json.load(open('intents.json','rb'))['intents']
    itoid = []
    phrase_embs = []
    for intent in intents:
        for phrase in intent['phrases']:
            itoid.append(phrase['intent_id'])
            phrase_embs.append(model(phrase['value']).numpy())
    phrase_arr = np.vstack(phrase_embs)
    return phrase_arr, itoid

def create_app():
    app = Flask(__name__)
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.secret_key = 'mick'
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(IntentClassifier, '/intent_classifier')


class IntentClassifier(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('value',
        type=str,
        required=True,
        help="Sentence to send to chatbot agent cannot be empty."
        )

    # Heroku crashed at this method
    @staticmethod
    def get_intent(sentence):
        sent_vec = model(sentence).numpy()
        sim_score = sent_vec @ phrase_arr.T
        return int(pd.DataFrame({'intent_id':itoid, 'score':sim_score.squeeze()}).groupby('intent_id').max().idxmax()['score'])

    def get(self):
        payload = self.__class__.parser.parse_args()
        intent_id = self.get_intent(payload['value'])
        return {'intent_id':intent_id}

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
phrase_arr, itoid = instantiate_reps_from_s3(bucket_name=BUCKET_NAME, fn='intents.json')


from db import db
from resources.intent import Intent, IntentList, PutToS3
from resources.response import Response
from resources.phrase import Phrase
from resources.chat import Chat
from resources.chat_line import LineChat

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL','sqlite:///data.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.secret_key = 'mick'

    db.init_app(app)
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(Intent, '/intent/<string:value>')
    api.add_resource(IntentList, '/intents')
    api.add_resource(Response, '/response')
    api.add_resource(Phrase, '/phrase')
    api.add_resource(Chat, '/chat')
    api.add_resource(LineChat, '/webhook')
    api.add_resource(PutToS3, '/put_to_s3')

app = create_app()
create_api(app)

@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == "__main__":
    app.run(port=5000, debug=True)
