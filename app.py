# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load trained model
# with open("model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# # Define default values for missing features
# default_values = {
#     "house_age": 30,  # Example default
#     "bed_bath_ratio": 1.2,
#     "total_rooms": 5
# }

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get input from form
#         user_input = request.json

#         # Convert inputs to float
#         YearBuilt = float(user_input["YearBuilt"])
#         BedroomAbvGr = float(user_input["BedroomAbvGr"])
#         FullBath = float(user_input["FullBath"])
#         HalfBath = float(user_input["HalfBath"])
#         KitchenAbvGr = float(user_input["KitchenAbvGr"])

#         # Compute engineered features
#         house_age = 2024 - YearBuilt
#         bed_bath_ratio = BedroomAbvGr / (FullBath + 1)
#         total_rooms = BedroomAbvGr + FullBath + HalfBath + KitchenAbvGr

#         # Construct full feature array
#         feature_vector = np.array([
#             house_age, bed_bath_ratio, total_rooms
#         ]).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(feature_vector)

#         return jsonify({"prediction": prediction.tolist()})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load trained model
# with open("model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# # Define all required features (replace with actual feature names from your dataset)
# feature_names = [
#     "YearBuilt", "BedroomAbvGr", "FullBath", "HalfBath", "KitchenAbvGr",
#     "house_age", "bed_bath_ratio", "total_rooms"  # Include all 78 feature names
# ]

# # Default values for missing features
# default_values = {
#     "house_age": 30,
#     "bed_bath_ratio": 1.2,
#     "total_rooms": 5,
#     # Add default values for all remaining features
# }

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get input data
#         user_input = request.json

#         # Prepare feature list
#         feature_vector = []
        
#         for feature in feature_names:
#             if feature in user_input:
#                 feature_vector.append(float(user_input[feature]))  # Convert user input to float
#             else:
#                 feature_vector.append(default_values.get(feature, 0))  # Use default value if missing

#         # Convert to NumPy array and reshape
#         feature_vector = np.array(feature_vector).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(feature_vector)

#         return jsonify({"prediction": prediction.tolist()})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# List all 78 expected features (copy from your model training script)
feature_names = [
    
    'Id','SubClass','Zoning','Frontage','Area','Street','Alley','Shape','Contour','Utilities','Config','Slope','Neighborhood',
    'Condition1','Condition2','BldgType','HouseStyle','Qual','Cond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st',
    'Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF',
    'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRAbvGrd',
    'Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
    'ScreenPorch','PoolArea','PoolQC','Fence','Feature','Val','MoSold','YrSold','SaleType','SaleCondition','house_age','bed_bath_ratio' ,'total_rooms'

    # Add all 78 feature names here
]

# Default values for missing features (set reasonable defaults)
default_values = {feature: 0 for feature in feature_names}  # Default all to 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        user_input = request.json

        # Prepare feature list
        feature_vector = []
        for feature in feature_names:
            if feature in user_input:
                feature_vector.append(float(user_input[feature]))  # Convert to float
            else:
                feature_vector.append(default_values[feature])  # Use default value

        import pandas as pd  # Ensure pandas is imported

        # Convert to Pandas DataFrame with column names
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

        # Make prediction
        prediction = model.predict(feature_df)


        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
