
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


feature_names = [
    
    'Id','SubClass','Zoning','Frontage','Area','Street','Alley','Shape','Contour','Utilities','Config','Slope','Neighborhood',
    'Condition1','Condition2','BldgType','HouseStyle','Qual','Cond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st',
    'Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF',
    'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRAbvGrd',
    'Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
    'ScreenPorch','PoolArea','PoolQC','Fence','Feature','Val','MoSold','YrSold','SaleType','SaleCondition','house_age','bed_bath_ratio' ,'total_rooms'


]

# Default values for missing features
default_values = {feature: 0 for feature in feature_names}  # Default all to 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        user_input = request.json

    
        feature_vector = []
        for feature in feature_names:
            if feature in user_input:
                feature_vector.append(float(user_input[feature]))
            else:
                feature_vector.append(default_values[feature])

        import pandas as pd  

       
        feature_df = pd.DataFrame([feature_vector], columns=feature_names)

 
        prediction = model.predict(feature_df)


        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
