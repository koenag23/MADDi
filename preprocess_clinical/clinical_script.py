import json

# Load the JSON file
input_file = "preprocess_clinical/clinical.json"
output_file = "preprocess_clinical/clinical_prompts.json"

#this one is for {NXABNORM}
def map_eligibility(value):
    return "Eligible" if value == "1.0" else "Ineligible"

#thsi one is for {PTGENDER}
def map_gender(value):
    return "Male" if value == "1.0" else "Female"

#this one is for {PTHAND}
def map_hand(value):
    return "Right" if value == "1.0" else "Left"

#this one is for {PTHOME}
def map_living_situation(value):
    return {"-4.0": "Information not available", "1.0": "House (owned or rented)", "2.0": "Condo/Co-op (owned)", "3.0": "Apartment (rented)", "4.0": "Mobile Home", "5.0": "Retirement Community", "6.0": "Assisted Living", "7.0": "Other"}.get(value, "Unknown")

#this one is for {PTMARRY}
def map_marital_status(value):
    return {"-4.0": "Information not available", "1.0": "Married", "2.0": "Widowed", "3.0": "Divorced", "4.0": "Never Married", "5.0": "Unknown"}.get(value, "Unknown")

#this one is for {PTPLANG}
def map_primary_language(value):
    return {"-4.0": "Information not available", "1.0": "English", "2.0": "Spanish", "3.0": "Other"}.get(value, "Unknown")

#this one is for {PTETHCAT}
def map_ethnicity(value):
    return {"-4.0": "Information not available", "1.0": "Hispanic or Latino", "2.0": "Not Hispanic or Latino", "3.0": "Unknown"}.get(value, "Unknown")

#this one is for {PTRACCAT} 
def map_race(value):
    return {"-4.0": "Information not available", "1.0": "American Indian or Alaskan Native", "2.0": "Asian", "3.0": "Native Hawaiian or Other Pacific Islander", "4.0": "Black or African American", "5.0": "White", "6.0": "More than one race", "7.0": "Unknown"}.get(value, "Unknown")

#this one is for {NXNERVE}, {NXCONSCI}, {NXMOTOR}, {NXPLANTA}, {NXGAIT}, {NXTENDON}
def map_normal_abnormal(value):
    if value == "-4.0":
        return "Information not available"
    return "Normal" if value == "1.0" else "Abnormal"

#this one is for {NXVISUAL}, {NXTREMOR}, {NXSENSOR}, {NXAUDITO}
def map_presence(value):
    if value == "-4.0":
        return "Information not available"
    return "Present" if value == "1.0" else "Absent"

# Read JSON data
with open(input_file, "r") as f:
    patients = json.load(f)

output_data = {}

# Process each patient
for patient_id, data in patients.items():
    present_symptoms = []
    absent_symptoms = []
    
    symptoms = {
        "Significant Vision Impairment": data["NXVISUAL"],
        "Tremors": data["NXTREMOR"],
        "Sensory Issues": data["NXSENSOR"],
        "Significant Auditory Impairment": data["NXAUDITO"]
    }
    
    for symptom, value in symptoms.items():
        if map_presence(value) == "Present":
            present_symptoms.append(symptom)
        else:
            absent_symptoms.append(symptom)
    
    present_sentence = "The symptoms that are present are: " + ", ".join(present_symptoms) + "." if present_symptoms else "No symptoms are present."
    absent_sentence = "The symptoms that are absent are: " + ", ".join(absent_symptoms) + "." if absent_symptoms else "No symptoms are absent."
    
    present_sentence = "The symptoms that are present are: " + ", ".join(present_symptoms) + "." if present_symptoms else "No symptoms are present."
    absent_sentence = "The symptoms that are absent are: " + ", ".join(absent_symptoms) + "." if absent_symptoms else "No symptoms are absent."
    
    question = f""" 
    Patient {patient_id} ({data['RID']}) is part of the {data['Phase']} phase.
    Based on a neurological examination, the patient was {map_eligibility(data['NXABNORM'])} for the study.
    The patient was assigned {map_gender(data['PTGENDER'])} at birth.
    They live in {map_living_situation(data['PTHOME'])} and are currently {map_marital_status(data['PTMARRY'])}.
    
    Their primary language is {map_primary_language(data['PTPLANG'])}.
    Their racial category is {map_race(data['PTRACCAT'])} and ethnicity is {map_ethnicity(data['PTETHCAT'])}.
    
    They were born in {data['PTDOBYY']} and were {data['AGE']} years old at their last visit on {data['VISDATE']}.
    
    {present_sentence}
    {absent_sentence}
    
    Their cranial nerves are {map_normal_abnormal(data['NXNERVE'])}, and their level of consciousness is {map_normal_abnormal(data['NXCONSCI'])}.
    Motor strength is {map_normal_abnormal(data['NXMOTOR'])}, plantar reflexes are {map_normal_abnormal(data['NXPLANTA'])}, gait is {map_normal_abnormal(data['NXGAIT'])}, and deep tendon reflexes are {map_normal_abnormal(data['NXTENDON'])}.
    
    Additional cognitive and physical health indicators include:
    - Harmonized composite language score: {data['PHC_LAN']}
    - Harmonized composite executive function score: {data['PHC_EXF']}
    - Harmonized composite visuospatial score: {data['PHC_VSP']}
    - Harmonized composite memory score: {data['PHC_MEM']}
    """

    
    output_data[patient_id] = {
        "question": question.strip(),
        "answer": data["GroupN"]
    }

# Save as JSON
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Patient reports saved to {output_file}")
