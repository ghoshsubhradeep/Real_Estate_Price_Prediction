import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

#method to predict price for real estate
def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x_ = np.zeros(len(__data_columns))
    x_[0] = sqft
    x_[1] = bath
    x_[2] = bhk
    if loc_index >= 0:
        x_[loc_index] = 1
    return round(__model.predict([x_])[0],2)

#method to get location names
def get_location_names():
    load_saved_artifacts()
    return __locations

#method to load required pickle file for model and json file for column values
def load_saved_artifacts():
    print("loading saved artifacts...start")
    #glov=balising variables
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open("./artifacts/bangalore_house_prices_model.pickle","rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


if __name__=="__main__":
    #sample run
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('Kalhalli',1000,2,2))    #other location
    print(get_estimated_price('Ejipura',1000,2,2))     #other location
    