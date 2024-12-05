import pandas as pd
import pickle
import numpy as np
import random
import warnings
import shutil

# Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# from pandas.core.common import SettingWithCopyWarning
 
 
def calculate_bmi(weight, height):
    """Calculate Body Mass Index (BMI)."""
    return weight / (height / 100) ** 2
 
def calculate_weight_boundaries(height, current_weight):
    """Calculate weight boundaries based on height and current weight."""
    height_m = height / 100  # Convert height to meters
    bmi_normal = 22  # Using the average BMI value for normal weight
    w_normal = bmi_normal * height_m ** 2
 
    # Upper boundary of weight
    w_upper = 24.9 * height_m ** 2
    w_over_min = current_weight - w_upper
 
    # Lower boundary of weight
    w_lower = 18.5 * height_m ** 2
    w_over_max = current_weight - w_lower
 
    return w_over_min, w_over_max
 
def calculate_bmr(gender, weight, height, age):
    """Calculate Basal Metabolic Rate (BMR)."""
    if gender.lower() == 'man':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender.lower() == 'woman':
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.33 * age)
    else:
        raise ValueError("Gender must be 'man' or 'woman'")
 
def calculate_daily_caloric_needs(bmr, activity_level):
    """Calculate Daily Needed Calories (DNC) based on activity level."""
    activity_multipliers = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }
 
    if activity_level.lower() in activity_multipliers:
        return bmr * activity_multipliers[activity_level.lower()]
    else:
        raise ValueError("Activity level must be one of the following: sedentary, lightly active, moderately active, very active, extra active")
 
def map_activity_level(activity_number):
    """Convert activity level number to string."""
    activity_map = {
        '1': 'sedentary',
        '2': 'lightly active',
        '3': 'moderately active',
        '4': 'very active',
        '5': 'extra active'
    }
    return activity_map.get(activity_number, None)
 
def map_gender(gender_number):
    """Convert gender number to string."""
    gender_map = {
        '1': 'man',
        '2': 'woman'
    }
    return gender_map.get(gender_number, None)
 
def load_model_and_data(model_path='./best_meal_plan.pkl'):
    try:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
            return data['meal_plan'], data['food_data']
    except FileNotFoundError:
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")
 
def filter_foods_based_on_preferences(food_df, preferences):
    final_filtered_df = pd.DataFrame()
    for meal_idx, attrs in enumerate(preferences):
        combined_mask = np.zeros(food_df.shape[0], dtype=bool)
        for attr in attrs:
            if attr in food_df.columns:
                combined_mask |= (food_df[attr] > 0)
        filtered_df = food_df[combined_mask]
        for attr in attrs:
            if attr in food_df.columns:
                pref_mask = food_df[attr] > 0
                if filtered_df[pref_mask].empty:
                    pref_food = food_df[pref_mask].head(1)
                    filtered_df = pd.concat([filtered_df, pref_food]).drop_duplicates()
        if not filtered_df.empty:
            filtered_df['Meal'] = f'Meal {meal_idx + 1}'
            final_filtered_df = pd.concat([final_filtered_df, filtered_df]).drop_duplicates().reset_index(drop=True)
    return final_filtered_df
 
def apply_preferences(food_df, preferences):
    filtered_df = filter_foods_based_on_preferences(food_df, preferences)
    return filtered_df
 
def random_walk(food_df, calorie_limit, max_iterations=100):
    best_individual = None
    best_fitness = np.inf
    iteration = 0
    
    while iteration < max_iterations:
        num_items_to_select = random.randint(1, min(10, len(food_df)))
        individual = random.sample(range(len(food_df)), num_items_to_select)
        
        meal_items = food_df.iloc[individual]
        total_calories = meal_items['Calories'].sum()
        
        fitness = abs(calorie_limit - total_calories)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_individual = individual
        
        if fitness <= 50:
            break
        
        iteration += 1
    
    return best_individual
 
def user_preferences(food_df, dnc_sat):
    non_preference_columns = ['Food_Code', 'Display_Name', 'Portion_Default', 'Portion_Amount', 'Portion_Display_Name', 'Calories']
    attributes = [col for col in food_df.columns if col not in non_preference_columns]
    

    # Number of columns
    print("Available attributes".center(shutil.get_terminal_size().columns))
    divider = "=" * 80
    print(divider.center(shutil.get_terminal_size().columns))
    columns = 3

    # Determine the width for each column
    column_width = 25

    # Loop through attributes and print in rows
    for idx, attr in enumerate(attributes):

        terminal_width = shutil.get_terminal_size().columns

        # Calculate total table width
        table_width = columns * column_width

        # Calculate left padding to center the table
        padding = (terminal_width - table_width) // 2
        # Print padding spaces at the start of each row
        if idx % columns == 0:
            print(" " * padding, end="")
        
        # Print each item left-aligned within its column
        print(f"{idx + 1}. {attr:<{column_width - len(str(idx + 1)) - 2}}", end="")

        # Add a new line after every 'columns' attributes
        if (idx + 1) % columns == 0:
            print()

    # Ensure the final row ends with a newline
    if len(attributes) % columns != 0:
        print()
    
    preferences = []
    print("\nPlease specify your preferences for each meal.")
    
    for i in range(1, 4):
        print(f"\nMeal {i} preferences:")
        attr_numbers = input(f"Enter the numbers of the attributes you want for meal {i} (comma-separated): ").strip().split(',')
        selected_attrs = [attributes[int(num) - 1].strip() for num in attr_numbers if num.isdigit()]
        preferences.append(selected_attrs)
    
    return preferences
 
def display_meal_plan(meal_items, meal_type):
    if meal_items is not None and not meal_items.empty:
        print(f"\nBest {meal_type} Plan:")
        for index, row in meal_items.iterrows():
            print(f"Food: {row['Display_Name']}, Portion: {row['Portion_Default']}, Calories: {row['Calories']:.2f}")
        print(f"Total Calories for {meal_type}: {meal_items['Calories'].sum():.2f}")
    else:
        print(f"No satisfactory {meal_type} plan found.")
 
def main():
    columns = shutil.get_terminal_size().columns

    # Multiline text for the Meal Planner App
    text1 = '''
    Welcome to the Meal Planner App!
    ▗▄▄▖ ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▄▖ 
    ▐▌ ▐▌  █  ▐▛▚▖▐▌  █  ▐▌ ▐▌
    ▐▛▀▘   █  ▐▌ ▝▜▌  █  ▐▌ ▐▌
    ▐▌   ▗▄█▄▖▐▌  ▐▌  █  ▝▚▄▞▘
    '''
    text2 = '''
    ▗▄▄▖ ▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖▗▖ ▗▖
    ▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌
    ▐▛▀▚▖▐▛▀▀▘ ▝▀▚▖▐▛▀▀▘▐▛▀▜▌▐▛▀▚▖▐▌   ▐▛▀▜▌
    ▐▌ ▐▌▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▐▌ ▐▌▐▌ ▐▌▝▚▄▄▖▐▌ ▐▌
    '''

    # Center each line
    for line in text1.split('\n'):
        print(line.center(columns))
    for line in text2.split('\n'):
        print(line.center(columns))
        
    try:
        print("Enter your gender (1 for man, 2 for woman)".center(columns))
        print(" " * (columns // 2 - 5) + "►►►  ", end="")
        # Read user input
        gender_input = input()
        print()
        gender = map_gender(gender_input)
        if gender is None:
            raise ValueError("Invalid gender number. Please enter 1 for man or 2 for woman.")
        
        print("Enter your height in cm".center(columns))
        print(" " * (columns // 2 - 5) + "►►►  ", end="")
        height = float(input())
        print()
        print("Enter your weight in kg".center(columns))
        print(" " * (columns // 2 - 5) + "►►►  ", end="")
        weight = float(input())
        print()
        print("Enter your age in years".center(columns))
        print(" " * (columns // 2 - 5) + "►►►  ", end="")
        age = int(input())
        print()
        
        print("Enter your activity level".center(columns))
        print("1. sedentary 2. lightly active 3. moderately active 4. very active 5. extra active".center(columns))
        print(" " * (columns // 2 - 5) + "►►►  ", end="")
        activity_input = input()
        print()
        activity_level = map_activity_level(activity_input)
        if activity_level is None:
            raise ValueError("Invalid activity level number. Please enter a number between 1 and 5.")
 
        bmi = calculate_bmi(weight, height)
        print(f"BMI: {bmi:.2f}")
 
        w_over_min, w_over_max = calculate_weight_boundaries(height, weight)
        print(f"W(over-min): {w_over_min:.2f}")
        print(f"W(over-max): {w_over_max:.2f}")
 
        bmr = calculate_bmr(gender, weight, height, age)
        print(f"BMR: {bmr:.2f} calories/day")
 
        dnc = calculate_daily_caloric_needs(bmr, activity_level)
        print(f"Daily Needed Calories (DNC): {dnc:.2f} calories/day")
 
        dnc_saturated = dnc - 500
        print(f"Calorie needs for the program (DNC saturated): {dnc_saturated:.2f} calories/day")
 
        meal_plan, food_data = load_model_and_data()
 
        preferences = user_preferences(food_data, dnc_saturated)
        filtered_foods_df = apply_preferences(food_data, preferences)
        
        if filtered_foods_df.empty:
            print("No foods match the selected preferences.")
            return
        
        # Calculate individual meal calorie limits (e.g., 3 meals)
        breakfast_calories = dnc_saturated * 0.1  # 10% of daily calories for breakfast
        lunch_calories = dnc_saturated * 0.4    # 40% for lunch
        dinner_calories = dnc_saturated * 0.5      # 50% for dinner
 
        print(f"Caloric Targets : {dnc_saturated:.2f} calories/day")
 
        # Dictionary to store meal plans
        meal_plans = {}
        while True:
        # Generate meal plans for breakfast, lunch, and dinner
            for meal_type, calorie_limit in [('Breakfast', breakfast_calories), ('Lunch', lunch_calories), ('Dinner', dinner_calories)]:
                print(f"\nGenerating {meal_type} plan...")
                best_individual = random_walk(filtered_foods_df, calorie_limit)
                if best_individual:
                    meal_items = filtered_foods_df.iloc[best_individual]
                    meal_plans[meal_type] = meal_items
                else:
                    meal_plans[meal_type] = None
 
            # Display all meal plans together at the end
            for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
                display_meal_plan(meal_plans[meal_type], meal_type)
            print("Total Calories for the day:", sum([meal_plans[meal_type]['Calories'].sum() for meal_type in meal_plans if meal_plans[meal_type] is not None]))
            repeat = input("\nDo you want to generate another meal plan? (y/n): ")
            if repeat != 'y':
                    break  
 
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")
 
if __name__ == "__main__":
    main()