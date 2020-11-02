import preprocess
import pickle
import time


def load_model():
    start = time.time()
    f = open('model_svm.pkl', 'rb')
    svm = pickle.load(f)
    end_svm = time.time() - start
    vector = pickle.load(open('fit_vector.pkl', 'rb'))
    end_vec = time.time() - start - end_svm
    f.close()

    return svm, vector, end_svm, end_vec


def prediction(input, svm, vector):
    text = [preprocess.clean_data(input)]
    vector_tfid = vector.transform(text)
    return preprocess.form_output(svm.predict(vector_tfid)[0])


svm, vector, end_svm, end_vec = load_model()
print(prediction("""Cân Điện Tử Mặt Kính Cường Lực Trong Suốt""",svm, vector), end_svm, end_vec)
