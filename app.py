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

# Módulo recomendador
from recommender import ImprovedRecommender

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para desencriptar pickle del recomendador
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ImprovedRecommender":
            module = "recommender"
        return super().find_class(module, name)

app = FastAPI(title="E-commerce ML Models API")

# Modelos de validación
class RecommenderInput(BaseModel):
    user_id: Optional[int] = None
    product_history: Optional[List[int]] = None
    n_recommendations: int = 5

class SalesPredictionInput(BaseModel):
    store: int
    dept: int
    date: str

# Manejador de modelos optimizado
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
        self._last_load = {}
        self._model_cache = {}
    def load_initial_models(self):
        """Carga inicial de modelos y datos"""
        try:
            logger.info("Loading initial models and data...")
            
            # Cargar recomendador y sus datos
            if os.path.exists('models/recommender/recommender.pkl'):
                with open('models/recommender/recommender.pkl', 'rb') as f:
                    self._recommender = CustomUnpickler(f).load()
                self._products_df = pd.read_pickle('models/recommender/products_df.pkl')
                self._interactions_df = pd.read_pickle('models/recommender/interactions_df.pkl')
                self._recommender.process_data(self._products_df, self._interactions_df)
                logger.info("Recommender system loaded successfully")
            else:
                logger.warning("Recommender model files not found")

            # Cargar clasificador
            if os.path.exists('models/classifier/ecommerce_classifier.h5'):
                self._classifier = tf.keras.models.load_model(
                    'models/classifier/ecommerce_classifier.h5',
                    compile=False
                )
                logger.info("Classifier loaded successfully")
            else:
                logger.warning("Classifier model file not found")

            # Cargar modelo de ventas y datos relacionados
            if os.path.exists('models/sales/sales_prediction_model.h5'):
                self._sales_model = tf.keras.models.load_model(
                    'models/sales/sales_prediction_model.h5',
                    compile=False
                )
                self._scaler = joblib.load('models/sales/scaler.pkl')
                self._train_data = pd.read_csv('data/train.csv')
                self._stores_data = pd.read_csv('data/stores.csv')
                self._features_data = pd.read_csv('data/features.csv')
                logger.info("Sales prediction model and data loaded successfully")
            else:
                logger.warning("Sales prediction model files not found")

        except Exception as e:
            logger.error(f"Error loading initial models: {str(e)}")
            raise    
    def _check_reload_needed(self, model_name: str, reload_interval: int = 3600) -> bool:
        """Verifica si un modelo necesita ser recargado basado en el tiempo"""
        last_load = self._last_load.get(model_name)
        if last_load is None:
            return True
        return (datetime.now() - last_load).total_seconds() > reload_interval

    @property
    def recommender(self):
        if self._recommender is None or self._check_reload_needed('recommender'):
            logger.info("Loading recommender system...")
            try:
                with open('models/recommender/recommender.pkl', 'rb') as f:
                    self._recommender = CustomUnpickler(f).load()

                 # Asegurarse de que se calculen las recomendaciones populares
                if not hasattr(self._recommender, 'popular_recommendations'):
                    logger.info("Computing popular recommendations after loading...")
                    products_df = pd.read_pickle('models/recommender/products_df.pkl')
                    interactions_df = pd.read_pickle('models/recommender/interactions_df.pkl')
                    self._recommender.process_data(products_df, interactions_df)
                self._last_load['recommender'] = datetime.now()
            except Exception as e:
                logger.error(f"Error loading recommender: {str(e)}")
                raise
        return self._recommender

    @property
    def classifier(self):
        if self._classifier is None or self._check_reload_needed('classifier'):
            logger.info("Loading image classifier...")
            try:
                self._classifier = tf.keras.models.load_model(
                    'models/classifier/ecommerce_classifier.h5',
                    compile=False
                )
                self._classifier.make_predict_function()  # Optimización para inferencia
                self._last_load['classifier'] = datetime.now()
            except Exception as e:
                logger.error(f"Error loading classifier: {str(e)}")
                raise
        return self._classifier

    @property
    def sales_model(self):
        if self._sales_model is None or self._check_reload_needed('sales_model'):
            logger.info("Loading sales prediction model...")
            try:
                self._sales_model = tf.keras.models.load_model(
                    'models/sales/sales_prediction_model.h5',
                    compile=False
                )
                self._last_load['sales_model'] = datetime.now()
            except Exception as e:
                logger.error(f"Error loading sales model: {str(e)}")
                raise
        return self._sales_model

    @property
    def scaler(self):
        if self._scaler is None:
            logger.info("Loading scaler...")
            try:
                self._scaler = joblib.load('models/sales/scaler.pkl')
            except Exception as e:
                logger.error(f"Error loading scaler: {str(e)}")
                raise
        return self._scaler

    @property
    def train_data(self):
        if self._train_data is None:
            logger.info("Loading training data...")
            try:
                self._train_data = pd.read_csv('data/train.csv')
                self._train_data['Date'] = pd.to_datetime(self._train_data['Date'])
            except Exception as e:
                logger.error(f"Error loading training data: {str(e)}")
                raise
        return self._train_data

    @property
    def stores_data(self):
        if self._stores_data is None:
            logger.info("Loading stores data...")
            try:
                self._stores_data = pd.read_csv('data/stores.csv')
            except Exception as e:
                logger.error(f"Error loading stores data: {str(e)}")
                raise
        return self._stores_data

    @property
    def features_data(self):
        if self._features_data is None:
            logger.info("Loading features data...")
            try:
                self._features_data = pd.read_csv('data/features.csv')
                self._features_data['Date'] = pd.to_datetime(self._features_data['Date'])
            except Exception as e:
                logger.error(f"Error loading features data: {str(e)}")
                raise
        return self._features_data

    def clear_cache(self):
        """Limpia la caché de modelos"""
        self._recommender = None
        self._classifier = None
        self._sales_model = None
        self._last_load = {}
        self._model_cache = {}

# Inicializar manejador de modelos
models = ModelManager()

# Endpoints
@app.get("/")
def home():
    """Endpoint de salud y estado"""
    return {
        "status": "healthy",
        "models_available": [
            "recommender_system",
            "image_classifier",
            "sales_predictor"
        ],
        "server_time": datetime.now().isoformat()
    }

@app.post("/recommend")
def get_recommendations(input_data: RecommenderInput):
    """Endpoint para obtener recomendaciones de productos"""
    try:
        logger.info(f"Processing recommendation request: {input_data}")
        # Verificar el estado del recomendador
        logger.info("Checking recommender state...")



        # Verificar los archivos necesarios
        data_files = [
            'models/recommender/recommender.pkl',
            'models/recommender/products_df.pkl',
            'models/recommender/interactions_df.pkl'
        ]
        for file_path in data_files:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=500, 
                    detail=f"Required file not found: {file_path}"
                )
                
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

        return {
            "recommendations": recommendations,
            "metadata": {
                "request_type": "user_based" if input_data.user_id else 
                              "history_based" if input_data.product_history else 
                              "popular",
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """Endpoint para clasificar imágenes de productos"""
    try:
        logger.info(f"Processing image classification request for file: {file.filename}")
        
        # Validar tipo de archivo
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )

        # Guardar archivo temporalmente
        temp_path = f"temp_{file.filename}"
        try:
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Procesar imagen
            img = image.load_img(temp_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x/255.0

            # Hacer predicción
            prediction = models.classifier.predict(x, verbose=0)
            class_names = ['jeans', 'sofa', 'tshirt', 'tv']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            return {
                "classification": {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(class_names, prediction[0])
                    }
                },
                "metadata": {
                    "filename": file.filename,
                    "file_size": len(content),
                    "timestamp": datetime.now().isoformat()
                }
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sales")
def predict_sales(input_data: SalesPredictionInput):
    """Endpoint para predecir ventas"""
    try:
        logger.info(f"Processing sales prediction request: {input_data}")
        
        # Validaciones
        if input_data.store not in models.stores_data['Store'].values:
            raise HTTPException(status_code=400, detail=f"Invalid store: {input_data.store}")
        
        if input_data.dept not in models.train_data['Dept'].unique():
            raise HTTPException(status_code=400, detail=f"Invalid department: {input_data.dept}")
        
        try:
            date = pd.to_datetime(input_data.date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
        
        if date not in models.features_data['Date'].values:
            raise HTTPException(
                status_code=400, 
                detail=f"No feature data available for date: {input_data.date}"
            )

        # Preparar secuencia
        X_sequence = prepare_sequence_for_prediction(
            store=input_data.store,
            dept=input_data.dept,
            date=date,
            train_data=models.train_data,
            stores_data=models.stores_data,
            features_data=models.features_data
        )

        # Obtener patrones históricos
        patterns = analyze_historical_patterns(
            models.train_data,
            input_data.store,
            input_data.dept,
            date
        )

        # Predicción base
        X_scaled = models.scaler.transform(X_sequence)
        base_prediction = float(models.sales_model.predict(X_scaled)[0][0])

        # Ajustar predicción
        adjusted_prediction = base_prediction * (
            1.0 + (patterns['dow_factor'] - 1.0) * 0.7
        ) * (
            1.0 + (patterns['month_factor'] - 1.0) * 0.5
        )

        # Combinar con tendencia reciente
        weight_model = 0.3
        weight_recent = 0.7
        final_prediction = (
            adjusted_prediction * weight_model +
            patterns['recent_trend'] * weight_recent
        )

        # Calcular límites
        lower_bound = patterns['recent_trend'] - 1.5 * patterns['recent_std']
        upper_bound = patterns['recent_trend'] + 1.5 * patterns['recent_std']
        final_prediction = float(np.clip(final_prediction, lower_bound, upper_bound))

        return {
            "predictions": {
                "base_prediction": float(base_prediction),
                "adjusted_prediction": float(adjusted_prediction),
                "final_prediction": float(final_prediction)
            },
            "patterns": {
                "day_of_week_factor": float(patterns['dow_factor']),
                "month_factor": float(patterns['month_factor']),
                "recent_trend": float(patterns['recent_trend'])
            },
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            },
            "metadata": {
                "store": input_data.store,
                "department": input_data.dept,
                "date": date.strftime('%Y-%m-%d'),
                "timestamp": datetime.now().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting sales: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
def get_models_status():
    """Endpoint para verificar el estado de los modelos"""
    try:
        return {
            "recommender": {
                "loaded": models._recommender is not None,
                "last_load": models._last_load.get('recommender')
            },
            "classifier": {
                "loaded": models._classifier is not None,
                "last_load": models._last_load.get('classifier')
            },
            "sales_model": {
                "loaded": models._sales_model is not None,
                "last_load": models._last_load.get('sales_model')
            },
            "data_loaded": {
                "train_data": models._train_data is not None,
                "stores_data": models._stores_data is not None,
                "features_data": models._features_data is not None
            },
            "server_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommender/status")
def get_recommender_status():
    """Endpoint para verificar el estado del recomendador"""
    try:
        recommender = models.recommender
        has_popular = hasattr(recommender, 'popular_recommendations')
        num_popular = len(recommender.popular_recommendations) if has_popular else 0
        
        return {
            "status": "loaded",
            "has_popular_recommendations": has_popular,
            "num_popular_recommendations": num_popular,
            "data_dir_exists": recommender.data_dir.exists(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/system/check")
def check_system():
    """Verifica la existencia de archivos necesarios y estructura del sistema"""
    try:
        files_to_check = {
            'recommender': [
                'models/recommender/recommender.pkl',
                'models/recommender/products_df.pkl',
                'models/recommender/interactions_df.pkl'
            ],
            'classifier': [
                'models/classifier/ecommerce_classifier.h5'
            ],
            'sales': [
                'models/sales/sales_prediction_model.h5',
                'models/sales/scaler.pkl',
                'data/train.csv',
                'data/stores.csv',
                'data/features.csv'
            ]
        }

        status = {}
        for model, paths in files_to_check.items():
            status[model] = {
                'files_exist': all(os.path.exists(p) for p in paths),
                'files_status': {
                    p: {
                        'exists': os.path.exists(p),
                        'size': os.path.getsize(p) if os.path.exists(p) else 0
                    } for p in paths
                }
            }

        return {
            'system_check': status,
            'current_working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/models/reload")
def reload_models():
    """Endpoint para recargar todos los modelos"""
    try:
        models.clear_cache()
        return {"status": "success", "message": "All models reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


models = ModelManager()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
