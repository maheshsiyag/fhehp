"""
In this file, we roughly split up a list of symptoms, taken from "./training.csv" file, avalaible
through: "https://github.com/anujdutt9/Disease-Prediction-from-Symptoms/tree/master/dataset"
into medical categories, in order to make the UI more plesant for the users.

Each variable contains a list of symptoms sthat can be pecific to a part of the body or to a list
of similar symptoms.
"""

DIGESTIVE_SYSTEM_SYPTOMS = {
    "Digestive_system_symptoms": [
        "stomach_pain",
        "acidity",
        "vomiting",
        "indigestion",
        "constipation",
        "abdominal_pain",
        "diarrhoea",
        "belly_pain",
        "nausea",
        "distention_of_abdomen",
        "stomach_bleeding",
        "pain_during_bowel_movements",
        "passage_of_gases",
        "brittle_nails",
        "red_spots_over_body",
        "swelling_of_stomach",
        "bloody_stool",
        "yellowish_skin",
        "irritation_in_anus",
        "pain_in_anal_region",
        "abnormal_menstruation",
    ]
}

SKIN_SYPTOMS = {
    "Skin_related_symptoms": [
        "itching",
        "skin_rash",
        "pus_filled_pimples",
        "blackheads",
        "scurving",
        "skin_peeling",
        "silver_like_dusting",
        "small_dents_in_nails",
        "inflammatory_nails",
        "blister",
        "red_sore_around_nose",
        "bruising",
        "yellow_crust_ooze",
        "dischromic_patches",
        "nodal_skin_eruptions",
    ]
}

ORL_SYMPTOMS = {
    "ORL_SYMPTOMS": [
        "loss_of_smell",
        "continuous_sneezing",
        "runny_nose",
        "patches_in_throat",
        "throat_irritation",
        "sinus_pressure",
        "enlarged_thyroid",
        "loss_of_balance",
        "unsteadiness",
        "dizziness",
        "spinning_movements",
    ]
}

THORAX_SYMPTOMS = {
    "THORAX_RELATED_SYMPTOMS": [
        "breathlessness",
        "chest_pain",
        "cough",
        "rusty_sputum",
        "phlegm",
        "mucoid_sputum",
        "congestion",
        "blood_in_sputum",
        "fast_heart_rate",
    ]
}

EYES_SYMPTOMS = {
    "Eyes_related_symptoms": [
        "sunken_eyes",
        "redness_of_eyes",
        "watering_from_eyes",
        "blurred_and_distorted_vision",
        "pain_behind_the_eyes",
        "visual_disturbances",
    ]
}

VASCULAR_LYMPHATIC_SYMPTOMS = {
    "VASCULAR_LYMPHATIC_SYMPTOMS": [
        "cold_hands_and_feets",
        "swollen_blood_vessels",
        "swollen_legs",
        "swelled_lymph_nodes",
        "palpitations",
        "prominent_veins_on_calf",
        "yellowing_of_eyes",
        "puffy_face_and_eyes",
        "fluid_overload",
        "fluid_overload.1",
        "swollen_extremeties",
    ]
}

UROLOGICAL_SYMPTOMS = {
    "UROLOGICAL_SYMPTOMS": [
        "burning_micturition",
        "spotting_urination",
        "yellow_urine",
        "bladder_discomfort",
        "foul_smell_of_urine",
        "continuous_feel_of_urine",
        "polyuria",
        "dark_urine",
    ]
}

MUSCULOSKELETAL_SYMPTOMS = {
    "MUSCULOSKELETAL_SYMPTOMS": [
        "joint_pain",
        "muscle_wasting",
        "muscle_pain",
        "muscle_weakness",
        "knee_pain",
        "stiff_neck",
        "swelling_joints",
        "movement_stiffness",
        "hip_joint_pain",
        "painful_walking",
        "weakness_of_one_body_side",
        "neck_pain",
        "back_pain",
        "weakness_in_limbs",
        "cramps",
    ]
}

FEELING_SYMPTOMS = {
    "FEELING_SYMPTOMS": [
        "anxiety",
        "restlessness",
        "lethargy",
        "mood_swings",
        "depression",
        "irritability",
        "lack_of_concentration",
        "fatigue",
        "malaise",
        "weight_gain",
        "increased_appetite",
        "weight_loss",
        "loss_of_appetite",
        "obesity",
        "excessive_hunger",
    ]
}

OTHER_SYMPTOMS = {
    "OTHER_SYMPTOMS": [
        "ulcers_on_tongue",
        "shivering",
        "chills",
        "irregular_sugar_level",
        "high_fever",
        "slurred_speech",
        "sweating",
        "internal_itching",
        "mild_fever",
        "toxic_look_(typhos)",
        "acute_liver_failure",
        "dehydration",
        "headache",
        "extra_marital_contacts",
        "drying_and_tingling_lips",
        "altered_sensorium",
    ]
}

PATIENT_HISTORY = {
    "PATIENT_HISTORY": [
        "family_history",
        "receiving_blood_transfusion",
        "receiving_unsterile_injections",
        "history_of_alcohol_consumption",
        "coma",
    ]
}

SYMPTOMS_LIST = [
    SKIN_SYPTOMS,
    EYES_SYMPTOMS,
    ORL_SYMPTOMS,
    THORAX_SYMPTOMS,
    DIGESTIVE_SYSTEM_SYPTOMS,
    UROLOGICAL_SYMPTOMS,
    VASCULAR_LYMPHATIC_SYMPTOMS,
    MUSCULOSKELETAL_SYMPTOMS,
    FEELING_SYMPTOMS,
    PATIENT_HISTORY,
    OTHER_SYMPTOMS,
]
