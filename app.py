from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import joblib
from tensorflow.keras.preprocessing import image
import os
import logging
from functools import lru_cache

# Módulo recomendador (se cargará bajo demanda)
from recommender import ImprovedRecommender

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para desencriptar pickle del recomendador
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Si el pickle busca __main__.ImprovedRecommender, redirige a recommender.ImprovedRecommender
        if module == "__main__" and name == "ImprovedRecommender":
            module = "recommender"
        return super().find_class(module, name)

app = FastAPI(title="E-commerce ML Models API")

# Input validation models
class RecommenderInput(BaseModel):
    user_id: Optional[int] = None
    product_history: Optional[List[int]] = None
    n_recommendations: int = 5

class SalesPredictionInput(BaseModel):
    store: int
    dept: int
    date: str

# Carga perezosa de modelos
class ModelManager:
    def __init__(self):
        self._recommender = None
        self._classifier = None
        self._sales_model = None
        self._scaler = None
        self._products_df = None
        self._interactions_df = None
        self._train_data = None
        self._stores_data = None
        self._features_data = None
        
    @property
    def recommender(self):
        if self._recommender is None:
            logger.info("Loading recommender system on demand...")
            with open('models/recommender/recommender.pkl', 'rb') as f:
                self._recommender = CustomUnpickler(f).load()
        return self._recommender
    
    @property
    def products_df(self):
        if self._products_df is None:
            logger.info("Loading products dataframe on demand...")
            self._products_df = pd.read_pickle('models/recommender/products_df.pkl')
        return self._products_df
    
    @property
    def interactions_df(self):
        if self._interactions_df is None:
            logger.info("Loading interactions dataframe on demand...")
            self._interactions_df = pd.read_pickle('models/recommender/interactions_df.pkl')
        return self._interactions_df
    
    @property
    def classifier(self):
        if self._classifier is None:
            logger.info("Loading image classifier on demand...")
            self._classifier = tf.keras.models.load_model('models/classifier/ecommerce_classifier.h5', compile=False)
        return self._classifier
        
    @property
    def class_names(self):
        return ['jeans', 'sofa', 'tshirt', 'tv']
    
    @property
    def sales_model(self):
        if self._sales_model is None:
            logger.info("Loading sales prediction model on demand...")
            self._sales_model = tf.keras.models.load_model('models/sales/sales_prediction_model.h5', compile=False)
        return self._sales_model
    
    @property
    def scaler(self):
        if self._scaler is None:
            logger.info("Loading scaler on demand...")
            self._scaler = joblib.load('models/sales/scaler.pkl')
        return self._scaler
    
    @property
    def train_data(self):
        if self._train_data is None:
            logger.info("Loading sales training data on demand...")
            self._train_data = pd.read_csv('data/train.csv')
            self._train_data['Date'] = pd.to_datetime(self._train_data['Date'])
        return self._train_data
    
    @property
    def stores_data(self):
        if self._stores_data is None:
            logger.info("Loading stores data on demand...")
            self._stores_data = pd.read_csv('data/stores.csv')
        return self._stores_data
    
    @property
    def features_data(self):
        if self._features_data is None:
            logger.info("Loading features data on demand...")
            self._features_data = pd.read_csv('data/features.csv')
            self._features_data['Date'] = pd.to_datetime(self._features_data['Date'])
        return self._features_data

# Inicializar manejador de modelos (sin cargar nada aún)
models = ModelManager()

def prepare_sequence_for_prediction(store, dept, date, train_data, stores_data, features_data):
    """
    Prepara una secuencia para predicción sin escalado previo.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    store_info = stores_data[stores_data['Store'] == store].iloc[0]
    date_features = features_data[
        (features_data['Store'] == store) &
        (features_data['Date'] == date)
    ].iloc[0]

    store_sales = train_data[train_data['Store'] == store]['Weekly_Sales']
    dept_sales = train_data[
        (train_data['Store'] == store) &
        (train_data['Dept'] == dept)
    ]['Weekly_Sales']

    store_avg = store_sales.mean()
    store_std = store_sales.std()
    dept_avg = dept_sales.mean()
    dept_std = dept_sales.std()

    features = {
        'Store': store,
        'Dept': dept,
        'Temperature': date_features['Temperature'],
        'Fuel_Price': date_features['Fuel_Price'],
        'CPI': date_features['CPI'],
        'Unemployment': date_features['Unemployment'],
        'MarkDown1': date_features.get('MarkDown1', 0) if not pd.isna(date_features.get('MarkDown1')) else 0,
        'MarkDown2': date_features.get('MarkDown2', 0) if not pd.isna(date_features.get('MarkDown2')) else 0,
        'MarkDown3': date_features.get('MarkDown3', 0) if not pd.isna(date_features.get('MarkDown3')) else 0,
        'MarkDown4': date_features.get('MarkDown4', 0) if not pd.isna(date_features.get('MarkDown4')) else 0,
        'MarkDown5': date_features.get('MarkDown5', 0) if not pd.isna(date_features.get('MarkDown5')) else 0,
        'Month': date.month,
        'Week': date.isocalendar()[1],
        'DayOfWeek': date.dayofweek,
        'IsHoliday': int(date_features['IsHoliday']),
        'Size': store_info['Size'],
        'Store_Avg_Sales': store_avg,
        'Store_Std_Sales': store_std,
        'Dept_Avg_Sales': dept_avg,
        'Dept_Std_Sales': dept_std,
        'IsSummer': int(date.month in [6, 7, 8]),
        'IsWinter': int(date.month in [12, 1, 2]),
        'IsWeekend': int(date.dayofweek in [5, 6]),
        'Type_A': int(store_info['Type'] == 'A'),
        'Type_B': int(store_info['Type'] == 'B')
    }

    return pd.DataFrame([features])

@lru_cache(maxsize=128)
def analyze_historical_patterns(store, dept, date_str):
    """
    Analiza patrones históricos para ajustar predicciones.
    Con caché para reducir procesamiento repetido.
    """
    target_date = pd.to_datetime(date_str)
    
    store_dept_data = models.train_data[
        (models.train_data['Store'] == store) &
        (models.train_data['Dept'] == dept)
    ].copy()

    store_dept_data['DayOfWeek'] = store_dept_data['Date'].dt.dayofweek
    recent_data = store_dept_data[store_dept_data['Date'] <= target_date].tail(12)
    dow_pattern = recent_data.groupby('DayOfWeek')['Weekly_Sales'].mean()
    dow_factor = dow_pattern.get(target_date.dayofweek, 1.0) / dow_pattern.mean() if not dow_pattern.empty else 1.0

    store_dept_data['Month'] = store_dept_data['Date'].dt.month
    month_pattern = store_dept_data.groupby('Month')['Weekly_Sales'].mean()
    month_factor = month_pattern.get(target_date.month, 1.0) / month_pattern.mean()
    month_factor = 1.0 + (month_factor - 1.0) * 0.5

    recent_weeks = [4, 8, 12]
    weights = [0.5, 0.3, 0.2]
    recent_trends = []

    for weeks, weight in zip(recent_weeks, weights):
        data = store_dept_data[
            (store_dept_data['Date'] < target_date) &
            (store_dept_data['Date'] >= target_date - pd.Timedelta(weeks=weeks))
        ]
        if not data.empty:
            recent_trends.append((data['Weekly_Sales'].mean(), weight))

    recent_trend = (
        sum(trend * weight for trend, weight in recent_trends) / 
        sum(weight for _, weight in recent_trends)
    ) if recent_trends else store_dept_data['Weekly_Sales'].mean()

    recent_std = store_dept_data[
        store_dept_data['Date'] >= target_date - pd.Timedelta(weeks=12)
    ]['Weekly_Sales'].std()
    
    if pd.isna(recent_std):
        recent_std = store_dept_data['Weekly_Sales'].std()

    return {
        'dow_factor': float(dow_factor),
        'month_factor': float(month_factor),
        'recent_trend': float(recent_trend),
        'recent_std': float(recent_std)
    }

# Endpoints
@app.get("/")
def home():
    return {
        "health_check": "OK",
        "models_available": [
            "recommender_system",
            "image_classifier",
            "sales_predictor"
        ]
    }

@app.post("/recommend")
def get_recommendations(input_data: RecommenderInput):
    """
    Endpoint para obtener recomendaciones de productos.
    """
    try:
        logger.info(f"Processing recommendation request: {input_data}")
        
        if input_data.user_id is not None:
            logger.info(f"Getting recommendations for user {input_data.user_id}")
            recommendations = models.recommender.get_recommendations(
                input_data.user_id,
                n_recommendations=input_data.n_recommendations
            )
        elif input_data.product_history is not None:
            logger.info(f"Getting recommendations from product history: {input_data.product_history}")
            recommendations = models.recommender.get_recommendations_from_history(
                input_data.product_history,
                n_recommendations=input_data.n_recommendations
            )
        else:
            logger.info("Getting popular recommendations")
            recommendations = models.recommender._get_popular_recommendations(
                n_recommendations=input_data.n_recommendations
            )

        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """
    Endpoint para clasificar imágenes de productos.
    """
    try:
        logger.info(f"Processing image classification request for file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process image
        img = image.load_img(temp_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0

        # Make prediction
        prediction = models.classifier.predict(x)
        predicted_class = models.class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Clean up
        os.remove(temp_path)

        logger.info(f"Image classified as {predicted_class} with confidence {confidence}")
        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sales")
def predict_sales(input_data: SalesPredictionInput):
    """
    Endpoint para predecir ventas.
    """
    try:
        logger.info(f"Processing sales prediction request: {input_data}")
        
        # Validate store
        if input_data.store not in models.stores_data['Store'].values:
            raise HTTPException(status_code=400, detail=f"Invalid store: {input_data.store}")
        
        # Validate department
        if input_data.dept not in models.train_data['Dept'].unique():
            raise HTTPException(status_code=400, detail=f"Invalid department: {input_data.dept}")
        
        date = pd.to_datetime(input_data.date)
        
        # Validate date
        if date not in models.features_data['Date'].values:
            raise HTTPException(
                status_code=400, 
                detail=f"No feature data available for date: {input_data.date}"
            )

        # Prepare sequence
        X_sequence = prepare_sequence_for_prediction(
            store=input_data.store,
            dept=input_data.dept,
            date=date,
            train_data=models.train_data,
            stores_data=models.stores_data,
            features_data=models.features_data
        )

        # Get historical patterns (con caché)
        patterns = analyze_historical_patterns(
            input_data.store,
            input_data.dept,
            input_data.date
        )

        # Make base prediction
        X_scaled = models.scaler.transform(X_sequence)
        base_prediction = float(models.sales_model.predict(X_scaled)[0][0])

        # Adjust prediction with patterns
        adjusted_prediction = base_prediction * (
            1.0 + (patterns['dow_factor'] - 1.0) * 0.7
        ) * (
            1.0 + (patterns['month_factor'] - 1.0) * 0.5
        )

        # Combine with recent trend
        weight_model = 0.3
        weight_recent = 0.7
        final_prediction = (
            adjusted_prediction * weight_model +
            patterns['recent_trend'] * weight_recent
        )

        # Calculate bounds
        lower_bound = patterns['recent_trend'] - 1.5 * patterns['recent_std']
        upper_bound = patterns['recent_trend'] + 1.5 * patterns['recent_std']
        final_prediction = float(np.clip(final_prediction, lower_bound, upper_bound))

        logger.info(f"Sales prediction complete. Final prediction: {final_prediction}")
        return {
            "base_prediction": base_prediction,
            "adjusted_prediction": adjusted_prediction,
            "final_prediction": final_prediction,
            "patterns": patterns,
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting sales: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/valid-data")
def get_valid_prediction_data():
    """
    Endpoint para obtener datos válidos para predicciones.
    """
    try:
        logger.info("Retrieving valid prediction data")
        valid_stores = models.stores_data['Store'].unique().tolist()
        valid_depts = models.train_data['Dept'].unique().tolist()
        date_range = {
            'min_date': models.features_data['Date'].min().strftime('%Y-%m-%d'),
            'max_date': models.features_data['Date'].max().strftime('%Y-%m-%d')
        }
        
        logger.info("Valid prediction data retrieved successfully")
        return {
            "valid_stores": valid_stores,
            "valid_departments": valid_depts,
            "date_range": date_range,
            "store_types": models.stores_data['Type'].unique().tolist(),
            "total_stores": len(valid_stores),
            "total_departments": len(valid_depts)
        }
    except Exception as e:
        logger.error(f"Error retrieving valid prediction data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
