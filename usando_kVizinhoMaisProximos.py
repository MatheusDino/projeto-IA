import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score

# Aqui carregamos a base bruta
# e transformamos ela em um DataFrame "df" para ficar mais fácil e leve de trabalhar.
# Lembrando que no caso do nosso grupo nossa base já passou por um preprocessamento PCA.
df = pd.read_csv("creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']


# Aqui nós pegamos a base bruta e em vez de normalizar ela, já passamos diretamente para um ndarray,
# desse jeito a gente consegue rodar a base bruta nos testes e treinos também pra comparar no fim.
X_brute = X.to_numpy()


# Normalização usando MinMaxScaler
# Basicamente quando chamamos o MinMaxScaler() estamos chamando a função matématica que já vimos em aula:
# (X - Xmin) / (Xmax - Xmin). 
# O fit_transform() já devolve a base em formato ndarray também para utilizarmos nos testes mais a frente.
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Após a normalização, pegamos esse ndarray e transformamos de volta em um DataFrame,
# e uma vez DataFrame podemos trazer de volta a csv também, assim exportando a base já normalizada para poder comparar com a bruta.
normal_df = pd.DataFrame(X_minmax, columns=X.columns)
normal_df.to_csv("C:/Users/Administrator/Desktop/projetoIA/base_normalizada.csv") # Se não tiver a base normalizada, só descomentar.


# Então a partir de agora nós vamos definir o 10-fold cross-validation. Mas antes precisamos pegar um pedaço da base para fazer isso.
# Usando o train_test_split do scikit_learn, nós damos os arrays da base e da Class de referência,
# dizemos o tamanho do teste, e também um valor para "embaralhar" a base antes de recortar o pedaço que vai ser usado em teste. 
# Escolhemos o valor olhando outros exemplos na internet.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Agora preparando a nossa validação de acordo com o solicitado, 10 folds (splits) de um mesmo conjunto (vindo do train_test_split),
# repetindo 5 vezes para cada fold, e cada fold com distribuição aleatória dos dados. 
# Escolhemos o mesmo valor de shuffle (distribuição aleatória) do random_state anterior.
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)


# Nos models apenas definimos qual os algoritmos vamos testar.
models = {
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7)
}


# Por fim, vamos montar tudo. Ao definir a função de evaluate models, nós passamos a base que desejamos avaliar junto com a Class de
# referência, os models (os algoritmos a serem testados), e o cross-validation já devidamente preparado. No fim a função vai devolver
# para nós os indicadores de Precisão, F1-Score, Recall e score de curvas AUC-ROC.
def evaluate_models(X, y, models, cv):
    results = {}
    start_time = time.time()
    # Percorre a lista de algoritmos que demos.
    for model_name, model in models.items():
        precision_scores = []
        f1_scores = []
        recall_scores = []
        roc_auc_scores = []
        
        #Para cada split, usa a configuração do train_test_split para treinar e testar.
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Treina o modelo.
            model.fit(X_train, y_train)
            
            # Faz previsões.
            y_pred = model.predict(X_test)
            
            # Calcula métricas.
            precision_scores.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            roc_auc_scores.append(roc_auc_score(y_test, y_pred))
        
        # Armazena resultados dos indicadores.
        results[model_name] = {
            'Precision': np.mean(precision_scores),
            'F1-Score': np.mean(f1_scores),
            'Recall': np.mean(recall_scores),
            'AUC-ROC': np.mean(roc_auc_scores),
        }
    elapsed_time = time.time() - start_time
    results['Elapsed Time'] = elapsed_time
    
    return results


# Primeiro avalia usando a base bruta.
results_brutebase = evaluate_models(X_brute, y, models, cv)

print("Resultados para a base bruta:")
for model_name, metrics in results_brutebase.items():
    print(f"{model_name}: {metrics}")


# Em seguida avalia usando a base normalizada.
results_minmax = evaluate_models(X_minmax, y, models, cv)

print("Resultados para a base normalizada (MinMaxScaler):")
for model_name, metrics in results_minmax.items():
    print(f"{model_name}: {metrics}")