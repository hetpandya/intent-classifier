# Intent Classifier

Code for Medium blog post [Creating Your own Intent Classifier](https://medium.com/analytics-vidhya/creating-your-own-intent-classifier-b86e000a4926).

### Requirements
`pip install wget tensorflow==1.5 pandas numpy keras`

### Training
For training, check: [intent_classification.ipynb](https://github.com/thehetpandya/intent-classifier/blob/main/intent_classification.ipynb)

### Inference
```
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
 
model = load_model('models/intents.h5')

with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)
  
nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
print(nlu.get_intent("is it cold in India right now"))
```
