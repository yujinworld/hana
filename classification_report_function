"""모델 평가하는 함수 매번 짜기 귀찮아서 이렇게 만들어 두었음."""
from sklearn.metrics import classification_report


def evaluate_model(model):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print("학습성능")
    print(classification_report(y_train, y_train_pred))
    print("일반화성능")
    print(classification_report(y_test, y_test_pred))
