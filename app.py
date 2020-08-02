from flask import Flask, request, json
from flask_restful import Resource, Api
from json import dumps
from flask_cors import CORS, cross_origin
from flask import jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
api = Api(app)
CORS(app, support_credentials=True)

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route("/login")

@cross_origin(supports_credentials=True)

def login():
  return jsonify({'success': 'ok'})

class Employees(Resource):
  def post(self):
    a=request.json["data"]
 
    field_list =['score', 'reviews', 'guest', 'bedroom', 'bed', 'bath', 'amentity_wifi',
       'amentity_dryer', 'amentity_essentials', 'amentity_hangers',
       'amentity_hair_dryer', 'amentity_washer', 'amentity_iron',
       'amentity_air_conditioning', 'score_cleanliness', 'score_accuracy',
       'score_communication', 'score_location', 'score_check_in',
       'score_value', 'is_host_verified', 'is_superhost', 'no_pets',
       'can_parties', 'can_smoking', 'amentity_elevator', 'amentity_kitchen',
       'amentity_cable_tv', 'amentity_tv', 'host_reviews',
       'host_response_rate', 'host_response_time',
       'amentity_laptop-friendly_workspace', 'amentity_first_aid_kit',
       'amentity_indoor_fireplace', 'amentity_heating',
       'amentity_carbon_monoxide_alarm', 'amentity_free_parking_on_premises',
       'amentity_breakfast', 'amentity_hot_tub', 'amentity_fire_extinguisher',
       'amentity_gym', 'amentity_pool', 'amentity_smoke_alarm',
       'amentity_free_street_parking', 'amentity_private_entrance',
       'amentity_room-darkening_shades', 'amentity_building_staff',
       'amentity_paid_parking_off_premises', 'amentity_high_chair',
       'amentity_bathtub', 'shared_bath', 'private_bath',
       'amentity_private_living_room', 'amentity_lock_on_bedroom_door',
       'is_dong_nai', 'is_binh_duong', 'is_ngoai_tinh', 'is_phu_nhuan',
       'is_tan_phu', 'is_binh_tan', 'is_quan_4', 'is_can_gio', 'is_nha_be',
       'is_quan_5', 'is_go_vap', 'is_binh_chanh', 'is_quan_12', 'is_tan_binh',
       'is_quan_3', 'is_quan_7', 'is_hoc_mon', 'is_quan_9', 'is_quan_8',
       'is_quan_6', 'is_quan_2', 'is_quan_10', 'is_quan_1', 'is_quan_11',
       'is_cu_chi', 'is_thu_duc', 'is_binh_thanh', 'is_entire_home',
       'is_private_room', 'is_hotel_room', 'is_shared_room']
    b = np.full([1,86],0)   

    for x in a:
      if x in field_list:
        idx= field_list.index(x)
        b[0][idx]=a[x]

    filename = 'model_clean.sav'
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(b)
    print(result.ravel() )

    cleaning_fee = result.ravel()[0]
    b = np.append(b, [[cleaning_fee]], axis = 1)
    filename = 'model_svc.sav'
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(b)
    print(result.ravel() )
    service_fee = result.ravel()[0]
    b = np.append(b, [[service_fee]], axis = 1)

    print(b)
    filename = 'model_price.sav'
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(b)
    print(result.ravel() )
    return result.ravel()[0]
    
    
api.add_resource(Employees, '/employees') # Route_1
if __name__ == '__main__':
     app.run(port='5002')