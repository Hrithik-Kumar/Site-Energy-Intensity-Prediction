import pickle
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'models/catb_tuned_final.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('site_eui')

@app.route('/predict', methods=['POST'])
def predict():
    building = request.get_json()
    
    df = feature_extract(building)
    df = df.to_dict(orient='records')[0]

    X = dv.transform([df])

    y_pred = model.predict(X)

    result = {
        'site-energy-intensity': float(y_pred)
    }

    return jsonify(result)

def feature_extract(df_all):

    df_all = pd.DataFrame.from_dict(df_all, orient='index').transpose()
    
    # extract new weather statistics from the building location weather features
    temp = [col for col in df_all.columns if 'temp' in col]

    df_all['min_temp'] = df_all[temp].min(axis=1)
    df_all['max_temp'] = df_all[temp].max(axis=1)
    df_all['avg_temp'] = df_all[temp].mean(axis=1)
    df_all['std_temp'] = df_all[temp].std(axis=1)
    df_all['skew_temp'] = df_all[temp].skew(axis=1)


    # by seasons
    temp = pd.Series([col for col in df_all.columns if 'temp' in col])

    winter_temp = temp[temp.apply(lambda x: ('january' in x or 'february' in x or 'december' in x))].values
    spring_temp = temp[temp.apply(lambda x: ('march' in x or 'april' in x or 'may' in x))].values
    summer_temp = temp[temp.apply(lambda x: ('june' in x or 'july' in x or 'august' in x))].values
    autumn_temp = temp[temp.apply(lambda x: ('september' in x or 'october' in x or 'november' in x))].values


    ### winter
    df_all['min_winter_temp'] = df_all[winter_temp].min(axis=1)
    df_all['max_winter_temp'] = df_all[winter_temp].max(axis=1)
    df_all['avg_winter_temp'] = df_all[winter_temp].mean(axis=1)
    df_all['std_winter_temp'] = df_all[winter_temp].std(axis=1)
    df_all['skew_winter_temp'] = df_all[winter_temp].skew(axis=1)
    ### spring
    df_all['min_spring_temp'] = df_all[spring_temp].min(axis=1)
    df_all['max_spring_temp'] = df_all[spring_temp].max(axis=1)
    df_all['avg_spring_temp'] = df_all[spring_temp].mean(axis=1)
    df_all['std_spring_temp'] = df_all[spring_temp].std(axis=1)
    df_all['skew_spring_temp'] = df_all[spring_temp].skew(axis=1)
    ### summer
    df_all['min_summer_temp'] = df_all[summer_temp].min(axis=1)
    df_all['max_summer_temp'] = df_all[summer_temp].max(axis=1)
    df_all['avg_summer_temp'] = df_all[summer_temp].mean(axis=1)
    df_all['std_summer_temp'] = df_all[summer_temp].max(axis=1)
    df_all['skew_summer_temp'] = df_all[summer_temp].max(axis=1)
    ## autumn
    df_all['min_autumn_temp'] = df_all[autumn_temp].min(axis=1)
    df_all['max_autumn_temp'] = df_all[autumn_temp].max(axis=1)
    df_all['avg_autumn_temp'] = df_all[autumn_temp].mean(axis=1)
    df_all['std_autumn_temp'] = df_all[autumn_temp].std(axis=1)
    df_all['skew_autumn_temp'] = df_all[autumn_temp].skew(axis=1)

    df_all['month_cooling_degree_days'] = df_all['cooling_degree_days']/12
    df_all['month_heating_degree_days'] = df_all['heating_degree_days']/12

    # total area
    df_all['building_area'] = df_all['floor_area'] * df_all['elevation']
    # rating energy by floor
    df_all['floor_energy_star_rating'] = df_all['energy_star_rating']/df_all['elevation']

    return df_all
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)