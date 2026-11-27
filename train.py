import mlflow
import mlflow.sklearn
import joblib

from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, log_loss, balanced_accuracy_score
'''
создает файл sqlite локально в папке и там будут храниться все метаданные
метаданные это гиперпараметры модели, названия экспериментов, результаты метрик
имена и версии моделей
'''

'''
следующая команда создает ufc-fight-prediction если еще нету или выбирает ее для экспериментов 
'''
artifact = joblib.load("models/ufc_logistic_regression.joblib")
model = artifact['model']
threshold = artifact['threshold']
X_test, y_test = joblib.load("data/test_data.joblib")


y_proba = model.predict_proba(X_test)[:,1]#для red corner
prediction = (y_proba >= threshold).astype(int)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ufc-fight-prediction")

with mlflow.start_run(run_name='ufc-logistic-regression'):
    # Автоматически вытаскиваем все параметры из загруженного файла
    # log_param(), не может принимать параметры через запятую поэтому отдельно надо вызывать
    #и log_param потому что сохраняется один параметр
    mlflow.log_params(model.get_params())
    mlflow.log_param('threshold', threshold)

    mlflow.log_metrics({
        'test_accuracy': accuracy_score(y_test, prediction),
        'test_recall': recall_score(y_test, prediction),
        'test_precision': precision_score(y_test, prediction),
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'test_f1_score': f1_score(y_test, prediction),
        'test_log_loss': log_loss(y_test, y_proba),
        'test_balanced_accuracy_score': balanced_accuracy_score(y_test, prediction)
    })
    #sk_model = сохраняет model, сохраняет в папку ml-runs и присваивает ему определенный id эксперимента   
    #артефакты - это тяжелые файлы, то есть сама обученная модель
    #artifact_path=model, указывает куда положить файлы модели, то есть имя подпапки куда положить файлы модели
    #registered_model_name='ufc-fight-prediction', берет модель и отправляет в регистр модели
    mlflow.sklearn.log_model(sk_model=model, artifact_path='model', registered_model_name='ufc-fight-prediction',  skops_trusted_types=["numpy.dtype"])

    mlflow.log_artifact('models/ufc_logistic_regression.joblib', artifact_path='inference')





