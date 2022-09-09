import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ktrain
from ktrain import text
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball']

train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=0
)

test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=0
)

x_train = train.data
y_train = train.target

x_test = test.data
y_test = test.target

model_name = 'distilbert-base-uncased'

trans = text.Transformer(model_name, maxlen=512)

train_data = trans.preprocess_train(x_train, y_train)
test_data = trans.preprocess_test(x_test, y_test)

model = trans.get_classifier()
learner = ktrain.get_learner(model, train_data=train_data, val_data=test_data, batch_size=16)
learner.fit_onecycle(1e-4, 1)
learner.validate(class_names=categories)

predictor = ktrain.get_predictor(learner.model, preproc=trans)
x = 'Jesus Christ in the central figure in Christianity, and Allah is the central figure in Islam.'
res = predictor.predict(x)
print(res)

