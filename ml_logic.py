from PIL import Image
import io
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def classify_body(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]

        pose = mp_pose.Pose(static_image_mode=True)
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if not results.pose_landmarks:
            return None, 0.0, "No body landmarks detected."

        lm = results.pose_landmarks.landmark

        def pixel_dist(p1, p2):
            a = np.array([lm[p1].x * img_w, lm[p1].y * img_h])
            b = np.array([lm[p2].x * img_w, lm[p2].y * img_h])
            return np.linalg.norm(a - b)

        # Extract key ratios
        shoulder_width = pixel_dist(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        hip_width = pixel_dist(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
        waist_width = (pixel_dist(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP) +
                       pixel_dist(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)) / 2
        full_height = pixel_dist(mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_ANKLE)

        # Calculate ratios
        ratios = {
            "shoulder_hip": shoulder_width / hip_width if hip_width != 0 else 0,
            "waist_hip": waist_width / hip_width if hip_width != 0 else 0,
            "shoulder_waist": shoulder_width / waist_width if waist_width != 0 else 0,
            "hip_shoulder": hip_width / shoulder_width if shoulder_width != 0 else 0,
        }

        # Scoring system
        scores = {
            "Inverted Triangle": 0,
            "Pear": 0,
            "Rectangle": 0,
            "Oval": 0,
        }

        # Apply heuristic scoring
        if ratios["shoulder_hip"] > 1.2: scores["Inverted Triangle"] += 1
        if ratios["shoulder_waist"] > 1.25: scores["Inverted Triangle"] += 1

        if ratios["hip_shoulder"] > 1.15: scores["Pear"] += 1
        if ratios["waist_hip"] < 0.95: scores["Pear"] += 1

        if abs(ratios["shoulder_hip"] - 1.0) < 0.1: scores["Rectangle"] += 1
        if abs(ratios["shoulder_waist"] - 1.0) < 0.1: scores["Rectangle"] += 1

        if full_height > 0 and (waist_width / full_height) > 0.33: scores["Oval"] += 1

        # Select best match
        best_type = max(scores, key=scores.get)
        confidence = round(scores[best_type] / 2.0, 2)

        explanation = f"""Ratios:
- Shoulder/Hip: {ratios['shoulder_hip']:.2f}
- Waist/Hip: {ratios['waist_hip']:.2f}
- Shoulder/Waist: {ratios['shoulder_waist']:.2f}

Classification Confidence: {confidence}
Based on relative proportions of body structure."""

        return best_type, confidence, explanation

    except Exception as e:
        return None, 0.0, f"Processing error: {str(e)}"


def get_bmi_category(height, weight):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"


def get_difficulty(age, bmi_category):
    if age > 50 or bmi_category in ["Overweight", "Obese"]:
        return "Beginner"
    elif age < 30 and bmi_category == "Normal":
        return "Advanced"
    else:
        return "Intermediate"


def generate_workout_plan(image_bytes, height, weight, age):
    body_type, confidence, explain = classify_body(image_bytes)
    bmi_category = get_bmi_category(height, weight)
    difficulty = get_difficulty(age, bmi_category)

    if not body_type:
        return "❌ Error: Could not detect body. Upload clear full-body image."

    plan = {
        "Beginner": [
            "Wall push-ups - 3x10",
            "Chair squats - 3x12",
            "Walking - 20 mins",
            "Seated rows - 2x15",
            "Stretch: Neck & calves"
        ],
        "Intermediate": [
            "Push-ups - 4x12",
            "Goblet squats - 3x15",
            "Deadlifts - 3x10",
            "Mountain climbers - 4x20",
            "Stretch: Shoulders & hamstrings"
        ],
        "Advanced": [
            "Weighted squats - 4x10",
            "Incline bench press - 4x8",
            "Burpees - 4x15",
            "Barbell row - 3x10",
            "Stretch: Deep yoga flow"
        ]
    }

    return f"""💪 Personalized Workout Plan
📌 Body Type: {body_type} (Confidence: {confidence})
📈 BMI Category: {bmi_category}
🎯 Difficulty Level: {difficulty}
📊 Height: {height} cm | Weight: {weight} kg | Age: {age}

🧠 Workout Plan:
- {chr(10).join(plan[difficulty])}

🔍 Explainability:
{explain}
"""


def generate_meal_plan(image_bytes, goal, height, weight, age):
    body_type, confidence, explain = classify_body(image_bytes)
    bmi_category = get_bmi_category(height, weight)
    difficulty = get_difficulty(age, bmi_category)

    if not body_type:
        return "❌ Error: Could not detect body. Upload clear full-body image."

    if goal == "cut":
        meals = {
            "Breakfast": "Oats + Egg whites + Green tea",
            "Lunch": "Grilled Chicken + Quinoa + Greens",
            "Snack": "Apple + Low-fat yogurt",
            "Dinner": "Tofu stir-fry + Mixed veggies"
        }
    else:
        meals = {
            "Breakfast": "Avocado Toast + Protein Shake",
            "Lunch": "Salmon Rice Bowl + Broccoli",
            "Snack": "Granola bar + Banana",
            "Dinner": "Chicken Pasta + Cheese + Olive oil"
        }

    return f"""🍽 {goal.capitalize()} Meal Plan
📌 Body Type: {body_type} (Confidence: {confidence})
📈 BMI Category: {bmi_category}
🎯 Activity Level: {difficulty}
📊 Height: {height} cm | Weight: {weight} kg | Age: {age}

🥣 Breakfast: {meals['Breakfast']}
🥗 Lunch: {meals['Lunch']}
🍎 Snack: {meals['Snack']}
🍛 Dinner: {meals['Dinner']}

🔍 Explainability:
{explain}
"""
