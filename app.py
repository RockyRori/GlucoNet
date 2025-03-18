import numpy as np
import pickle
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

from RF import clf

# Define a flask app
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

print('Model loaded. Start serving...')

print('Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # This function is called when a user visits the website.
    # It returns the HTML content of the main page.
    # The main page is defined in the file templates/index.html.
    # The @app.route() decorator is a way to associate this function
    # with a particular URL. In this case, when the user visits
    # http://localhost:5000/, this function is called.
    # The "GET" argument specifies that this function should be
    # called when the user makes a GET request (i.e. when they
    # visit the website in their browser).
    # The "methods" argument is a list of all the HTTP methods
    # that this function should respond to. In this case, we
    # only want to respond to GET requests, so we only include
    # "GET" in the list.
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def get_data():
    """
    This function is called when a user visits the website
    (i.e. when they make a GET request to the root URL).
    It is also called when the user submits a form on the website
    (i.e. when they make a POST request to the root URL).
    """
    # Check if the request is a POST request
    if request.method == 'POST':
        """
        If the request is a POST request, then we know that the user
        has submitted a form on the website. We can access the data
        from the form using the request.form dictionary.
        """
        # Get the values from the form
        age = request.form['age']
        gender = request.form['gender']
        Polyuria = request.form['Polyuria']
        Polydipsia = request.form['Polydipsia']
        Weight = request.form['Weight']
        Weakness = request.form['Weakness']
        Polyphagia = request.form['Polyphagia']
        Thrush = request.form['Thrush']
        Blurring = request.form['Blurring']
        Itching = request.form['Itching']
        Irritability = request.form['Irritability']
        Healing = request.form['Healing']
        Paresis = request.form['Paresis']
        Stiffness = request.form['Stiffness']
        Alopecia = request.form['Alopecia']
        Obesity = request.form['Obesity']

        # Create a new list with the values from the form
        newpat = [
            [age, gender, Polyuria, Polydipsia, Weight, Weakness, Polyphagia, Thrush, Blurring, Itching, Irritability,
             Healing, Paresis, Stiffness, Alopecia, Obesity]]

        # Use the model to make a prediction based on the values
        result = model.predict(newpat)

        # If the prediction is 1, then the user has diabetes
        if result == 1:
            val = "Diabetes"
        # Otherwise, the user does not have diabetes
        else:
            val = "No Diabetes"

    # Render the HTML template with the result
    return render_template('index.html', value=val)


if __name__ == '__main__':
    app.run(debug=True)
