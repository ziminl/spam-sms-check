from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the value of the input field named 'username'
        text = request.form.get('text')
        return f'{text}'
        get_predictions(text)
    # Render the HTML form
    return '''
        <form method="POST">
            <input type="text" name="username" placeholder="Enter your username">
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    app.run()
