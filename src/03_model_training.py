import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path, PosixPath


def train_model(data_path: PosixPath):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        stratify=labels)

    # Train the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predict and compute accuracy
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print(f'{score * 100:.2f}% of samples were classified correctly!')

    return model


def save_model(model: RandomForestClassifier, model_path: PosixPath):
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)


if __name__ == '__main__':
    data_path = Path.cwd().parent / 'data'
    model_path = Path.cwd().parent / 'model.pkl'
    model = train_model(data_path=data_path)
    save_model(model, model_path=model_path)
