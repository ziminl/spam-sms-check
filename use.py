#test.py
#result = model.evaluate(X_test, y_test)
#loss = result[0]
#accuracy = result[1]
#precision = result[2]
#recall = result[3]
#print(f"Accuracy: {accuracy*100:.2f}%")
#print(f"Precision:   {precision*100:.2f}%")
#print(f"Recall:   {recall*100:.2f}%")


def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]


text = "o text or x text"
get_predictions(text)
