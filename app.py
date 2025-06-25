from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import gdown
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



def download_model(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet = False)

download_model("https://drive.google.com/uc?id=1ph-h84xsBD1gfSzJ1CfNMnVwMaycOZOl", "Diet_Recommend.pkl")
download_model("https://drive.google.com/uc?id=12UGD-_PlKRNpTj9Qt4iULwcO9jFJ-9rH", "Workout_Recommend.pkl")
download_model("https://drive.google.com/uc?id=1CiX1aTGPuZI4URHwHJIiWYhB9ImMeGZx", "scaler.pkl")
download_model("https://drive.google.com/uc?id=1liHA-mBYmADoXOPB-9vDRpnP81ObnJcO", "workout_scaler.pkl")


model = joblib.load('Diet_Recommend.pkl')
scaler = joblib.load('scaler.pkl')
model_w = joblib.load('Workout_Recommend.pkl')
scaler_w = joblib.load('workout_scaler.pkl')

diet_mapping = {
    "Diet_1": '<b>Salad:</b> Broccoli, Carrots, Spinach, Lettuce, Onion <br> <b>Protein Intake:</b> Cheese, Cattoge cheese, Skim Milk, Law fat Milk, and Baru Nuts <br> <b>Juice:</b> Fruit Juice, Aloe vera juice, Cold-pressed juice, and Watermelon juice',
    "Diet_2": '<b>Salad:</b> Garlic, Mushroom, Green Papper, Icebetg Lettuce <br> <b>Protein Intake:</b> Baru Nuts, Beech Nuts, Hemp Seeds, Cheese Spandwich <br> <b>Juice:</b> Apple Juice, Mango juice,and Beetroot juice',
    "Diet_3": '<b>Salad:</b> Garlic, Roma Tomatoes, Capers and Iceberg Lettuce <br> <b>Protein Intake:</b> Cheese Standwish, Baru Nuts, Beech Nuts, Squash Seeds, and Mixed Teff <br> <b>Juice:</b> Apple juice, beetroot juice and mango juice',
    "Diet_4": '<b>Salad:</b> Garlic, Roma Tomatoes, Capers, Green Papper, and Iceberg Lettuce <br> <b>Protein Intake:</b> Cheese Sandwich, Baru Nuts, Beech Nuts, Squash Seeds, Mixed Teff, peanut butter, and jelly sandwich <br> <b>Juice:</b> Apple juice, beetroot juice, and mango juice',
    "Diet_5": '<b>Salad:</b> Garlic, mushroon, green papper and water chestnut  <br> <b>Protein Intake:</b>  Baru Nuts, Beech Nuts, and black walnut <br><b>Juice:</b> Apple juice, Mango, and Beetroot Juice',
    "Diet_6": '<b>Salad:</b> Garlic, mushroon, green papper <br> <b>Protein Intake:</b>  Baru Nuts, Beech Nuts, and Hemp Seeds <br> <b>Juice:</b> Apple juice, Mango, and Beetroot Juice',
    "Diet_7": '<b>Salad:</b> Mixed greens, cherry tomatoes, cucumbers, bell peppers, carrots, celery <br> <b>Protein Intake:</b> Chicken, fish, tofu, or legumes <br> <b>Juice:</b> Green juice, kale juice, spinach juice, cucumber juice and apple juice',
    "Diet_8": '<b>Salad:</b> Tomatoes, Garlic, leafy greens, broccoli, carrots, and bell peppers <br> <b>Protein Intake:</b> poultry, fish, tofu, legumes, and low-fat dairy products <br> <b>Juice:</b> Apple juice, beetroot juice and mango juice'
}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/workout_tut')
def workout_tut():
    return render_template('workout_tut.html')

@app.route('/diet_workout', methods=['GET'])
def diet_workout():
    return render_template('diet_workout.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.form

    sex = 1 if data['sex'] == '1' else 0
    age = int(data['age'])
    hypertension = 1 if data['hypertension'] == '1' else 0
    diabetes = 1 if data['diabetes'] == '1' else 0
    bmi = float(data['bmi'])
    fitness_goal = 1 if data['fitness_goal'] == '1' else 0
    fitness_type = 1 if data['fitness_type'] == '1' else 0
    bodysize = data['bodysize']

    bodysize_encoded = [0, 0, 0, 0] 
    if bodysize == 'level_normal':
        bodysize_encoded[0] = 1
    elif bodysize == 'level_obese':
        bodysize_encoded[1] = 1
    elif bodysize == 'level_overweight':
        bodysize_encoded[2] = 1
    elif bodysize == 'level_underweight':
        bodysize_encoded[3] = 1

    features = [sex, age, hypertension, diabetes, bmi, fitness_goal, fitness_type] + bodysize_encoded

    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)
    
    recommended_diet = diet_mapping.get(prediction[0], "Unknown Diet")
    
    return render_template('diet_workout.html', recommended_diet=recommended_diet)

workout_map = {
    'Exercises_1' : '<b><h2>Upper-Lower</h2></b><br><table><tr><th>Day</th><th>Muscle Groups</th><th>Exercises</th></tr><tr><td>Day 1: Upper</td><td>Chest, Back, Shoulders, Arms</td><td>- Bench Press<br>- Pull-Ups<br>- Overhead Press<br>- Barbell Curls<br>- Tricep Dips</td></tr><tr><td>Day 2: Lower</td><td>Quads, Hamstrings, Glutes, Calves</td><td>- Squats<br>- Romanian Deadlifts<br>- Leg Press<br>- Bulgarian Split Squats<br>- Standing Calf Raises</td></tr><tr><td>Day 3: Upper</td><td>Chest, Back, Shoulders, Arms</td><td>- Incline Dumbbell Bench Press<br>- Dumbbell Rows<br>- Arnold Press<br>- Hammer Curls<br>- Tricep Pushdowns</td></tr><tr><td>Day 4: Lower</td><td>Quads, Hamstrings, Glutes, Calves</td><td>- Deadlifts<br>- Front Squats<br>- Walking Lunges<br>- Hamstring Curls<br>- Seated Calf Raises</td></tr></table>',
    'Exercises_2' : '<b><h2>Push-Pull-Legs (PPL)</b><br></h2><table><tr><th>Day</th><th>Muscle Groups</th><th>Exercises</th></tr><tr><td>Day 1: Push</td><td>Chest, Shoulders, Triceps</td><td>- Bench Press (Flat/Incline)<br>- Overhead Press<br>- Dips<br>- Tricep Pushdowns<br>- Dumbbell Lateral Raises</td></tr><tr><td>Day 2: Pull</td><td>Back, Biceps</td><td>- Deadlifts<br>- Pull-Ups<br>- Barbell Rows<br>- Face Pulls<br>- Barbell Curls</td></tr><tr><td>Day 3: Legs</td><td>Quads, Hamstrings, Glutes, Calves</td><td>- Squats<br>- Romanian Deadlifts<br>- Leg Press<br>- Lunges<br>- Standing Calf Raises</td></tr></table>',
    'Exercises_3' : '<b><h2>Push-Pull (2-Day)</h2></b><br><table><tr><th>Day</th><th>Muscle Groups</th><th>Exercises</th></tr><tr><td>Day 1: Push</td><td>Chest, Shoulders, Triceps</td><td>- Bench Press<br>- Overhead Press<br>- Dumbbell Fly<br>- Tricep Dips<br>- Dumbbell Lateral Raises</td></tr><tr><td>Day 2: Pull</td><td>Back, Biceps</td><td>- Deadlifts<br>- Pull-Ups<br>- Barbell Rows<br>- Face Pulls<br>- Hammer Curls</td></tr></table>',
    'Exercises_4' : '<b><h2>Bro Split</h2></b><br><table><tr><th>Day</th><th>Muscle Groups</th><th>Exercises</th></tr><tr><td>Day 1: Chest</td><td>Chest</td><td>- Bench Press<br>- Incline Dumbbell Press<br>- Chest Fly<br>- Push-Ups</td></tr><tr><td>Day 2: Back</td><td>Back</td><td>- Deadlifts<br>- Pull-Ups<br>- Barbell Rows<br>- Face Pulls</td></tr><tr><td>Day 3: Shoulders</td><td>Shoulders</td><td>- Overhead Press<br>- Dumbbell Lateral Raises<br>- Rear Delt Fly<br>- Dumbbell Shrugs</td></tr><tr><td>Day 4: Arms</td><td>Biceps, Triceps</td><td>- Barbell Curls<br>- Hammer Curls<br>- Tricep Pushdowns<br>- Skull Crushers</td></tr><tr><td>Day 5: Legs</td><td>Quads, Hamstrings, Glutes, Calves</td><td>- Squats<br>- Romanian Deadlifts<br>- Leg Press<br>- Lunges<br>- Standing Calf Raises</td></tr></table>',
    'Exercises_5' : '<b><h2>Full Body</h2></b><br><table><tr><th>Day</th><th>Muscle Groups</th><th>Exercises</th></tr><tr><td>Day 1</td><td>All Muscle Groups</td><td>- Deadlifts<br>- Bench Press<br>- Barbell Rows<br>- Overhead Press<br>- Squats</td></tr><tr><td>Day 2</td><td>All Muscle Groups</td><td>- Pull-Ups<br>- Push-Ups<br>- Bulgarian Split Squats<br>- Dumbbell Shoulder Press<br>- Leg Curls</td></tr><tr><td>Day 3</td><td>All Muscle Groups</td><td>- Romanian Deadlifts<br>- Incline Dumbbell Bench Press<br>- One-Arm Dumbbell Rows<br>- Lateral Raises<br>- Calf Raises</td></tr></table>'
}

@app.route('/predict_w', methods=['POST'])
def predict_w():
    data = request.form

    sex = 1 if data['sex'] == '1' else 0
    age = int(data['age'])
    diabetes = 1 if data['diabetes'] == '1' else 0
    bmi = float(data['bmi'])
    fitness_goal = 1 if data['fitness_goal'] == '1' else 0
    fitness_type = 1 if data['fitness_type'] == '1' else 0
    bodysize = data['bodysize']

    bodysize_encoded = [0, 0, 0, 0] 
    if bodysize == 'level_normal':
        bodysize_encoded[0] = 1
    elif bodysize == 'level_obese':
        bodysize_encoded[1] = 1
    elif bodysize == 'level_overweight':
        bodysize_encoded[2] = 1
    elif bodysize == 'level_underweight':
        bodysize_encoded[3] = 1

    features = [sex, age, diabetes, bmi, fitness_goal, fitness_type] + bodysize_encoded
    features = np.array(features).reshape(1, -1)

    features = scaler_w.transform(features)
    prediction = model_w.predict(features)

    print("Prediction:", prediction) 
    recommended_workout = workout_map.get(prediction[0], "Unknown Workout")
    
    return render_template('diet_workout.html', recommended_workout=recommended_workout)

nutrition_df = pd.read_csv("nutrition_data.csv")

def get_nutrition_info(food):
    food = food.lower()
    result = nutrition_df[nutrition_df['food_name'].str.contains(food, flags=re.IGNORECASE, regex=True)]
    if not result.empty:
        nutrition = result.iloc[0]
        response = (f"Nutrition values for {nutrition['food_name']}:\n"
                    f"- Calories: {nutrition['calories']} kcal\n"
                    f"- Protein: {nutrition['protein']}\n"
                    f"- Carbs: {nutrition['carbs']}\n"
                    f"- Fats: {nutrition['fats']}")
        return response
    else:
        return "Sorry, I don't have information about that food."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_nutrition_info(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

