from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from TaxiFareModel.params import MLFLOW_URI, EXPERIMENT_NAME
from TaxiFareModel.params import PATH_TO_LOCAL_MODEL
from TaxiFareModel.data import upload_model_to_gcp
from xgboost import XGBRegressor

class Trainer():

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        self.experiment_name = EXPERIMENT_NAME

        #self.mlflow_run()
        self.mlflow_log_param("student_name", "Geoffroy")

        self.kwargs = kwargs

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        if self.kwargs["estimator"] == "xgboost":
            pipe = Pipeline([
                ('preproc', preproc_pipe),
                ('model', XGBRegressor())
            ])
        else:
            pipe = Pipeline([
                ('preproc', preproc_pipe),
                ('model', LinearRegression())
            ])
        # add model type to MLflow
        self.mlflow_log_param("model", type(pipe["model"]))
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        # add RMSE to MLflow
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    def save_model(self, ):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, PATH_TO_LOCAL_MODEL)
        # and upload the joblib file to GCP
        upload_model_to_gcp(PATH_TO_LOCAL_MODEL)


if __name__ == "__main__":

    params = dict(nrows=10000,
              upload=True,
              local=False,  # set to False to get data from GCP (Storage or BigQuery)
              gridsearch=False,
              optimize=True,
              estimator="xgboost",
              mlflow=True,  # set to True to log params to mlflow
              #experiment_name=experiment,
              pipeline_memory=None, # None if no caching and True if caching expected
              distance_type="manhattan",
              feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"],
              n_jobs=-1) # Try with njobs=1 and njobs = -1

    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train, y_train, **params)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_val, y_val)
    print(rmse)
    # save
    trainer.save_model()
