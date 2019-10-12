import pandas as pd
from keras.models import model_from_json
from keras.utils import to_categorical

#Load the Dataset
data_path = "path/to/folder/containing/the/dataset"
df_test = pd.read_csv(data_path + "fashion-mnist_test.csv")

#Pre-processing to fit my model architecture
df_x = df_test.drop(labels = 'label', axis = 1)
df_y = df_test['label']

x_test = df_x.to_numpy()
x_test = x_test.reshape([10000,28,28,1])

y = df_y.to_numpy()
y_test = to_categorical(y,num_classes = 10)

#Load the model along with the weights
json_file = open('network.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights.h5")

# evaluate the model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test)
print(score)
