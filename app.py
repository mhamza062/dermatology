import streamlit as st
import joblib
import numpy as np

st.sidebar.title("About the App")
st.sidebar.markdown("This app predicts skin diseases based on symptoms.")

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title and description
st.title("🩺 Skin Disease Classifier (Patient-Friendly Form)")

st.markdown("""
This tool helps identify possible skin diseases based on symptoms. Please answer the following questions.
""")

# Mapping functions for binary and numeric questions
def binary_question(question):
    return 1 if st.radio(question, ["Yes", "No"]) == "Yes" else 0

def numeric_question(question, min_val=0, max_val=100):
    return st.slider(question, min_val, max_val, step=1)

erythema = binary_question("Kya aap ki skin laal ho jati hai ya us par surkhi rehti hai?")
scaling = binary_question("Kya aap ki skin chilti hai ya us par sukhapan se jhilmilahat hoti hai?")
definite_borders = binary_question("Kya skin ke daag ya nishan ek seedhi line mein ya clearly nazar aate hain?")
itching = binary_question("Kya aap ko kisi jagah par bar bar khujli mehsoos hoti hai?")
koebner_phenomenon = binary_question("Kya zakhm ya chot lagne ke baad wahan bhi naye daag ya nishan ban jaate hain?")
polygonal_papules = binary_question("Kya aap ke jism par chhoti chhoti, thodi si chaurasi (polygon shape) danay hain?")
follicular_papules = binary_question("Kya baalon ke aas paas chhoti chhoti danay nikal aate hain?")
knee_elbow_involvement = binary_question("Kya aap ke ghutne ya kohniyon par bhi daag ya danay hain?")
scalp_involvement = binary_question("Kya aap ke sir ki skin bhi is maslay ka shikar hai?")
family_history = binary_question("Kya aap ke ghar walon mein kisi ko aisi skin ki bimari hui hai?")
eosinophils_in_infiltrate = binary_question("Kya doctors ne blood ya skin test mein allergy ya sujan wali cells batayi hain?")
PNL_infiltrate = binary_question("Kya doctors ne test mein kuch white blood cells ka ikattha hona bataya hai?")
fibrosis_of_the_papillary_dermis = binary_question("Kya skin ke neeche wali layer zyada hard ho gayi hai?")
exocytosis = binary_question("Kya aap ke skin test mein sujan wali cells skin ke andar dikhayi gayi hain?")
acanthosis = binary_question("Kya aap ki skin thodi mooti ya hard lagti hai?")
hyperkeratosis = binary_question("Kya aap ki skin par zyada sakht sukhapan ya jhilli type layer ban gayi hai?")
parakeratosis = binary_question("Kya skin ke upar wali layer thodi ajeeb si ya cell wali lagti hai (jaise chilka)?")
club_rete_ridges = binary_question("Kya skin ka neechay ka hissa thoda bulbula ya club shape ka lagta hai?")
elongation_of_the_rete_ridges = binary_question("Kya skin ka neeche ka texture lamba lamba sa lagta hai?")
spongiform_pustule = binary_question("Kya aap ke daanon mein peep bhari hui nazar aati hai?")
munro_microabcess = binary_question("Kya aap ke skin mein choti choti peep wali jagah hoti hain?")
disappearance_granular_layer = binary_question("Kya skin ka ek layer aisa lagta hai jaise gayab ho gaya ho?")
spongiosis = binary_question("Kya aap ki skin cells ke beech fulaav ya pani sa mehsoos hota hai?")
follicular_horn_plug = binary_question("Kya baalon ke raste mein kuch hard aur sukhay se dhagge jese cheez hoti hai?")
inflammatory_monoluclear_infiltrate = binary_question("Kya doctor ne skin test mein sujan wali cells ka hona bataya hai?")
age = numeric_question("Aap ki age kya hai (umar)?", 0, 100)


# Combine all inputs into a numpy array
input_data = np.array([[ 
    age, itching, scaling, definite_borders, erythema, 
    koebner_phenomenon, family_history, follicular_papules,
    inflammatory_monoluclear_infiltrate,
    fibrosis_of_the_papillary_dermis, elongation_of_the_rete_ridges,
     spongiform_pustule,
    munro_microabcess, parakeratosis,
    club_rete_ridges, 
     exocytosis, acanthosis, spongiosis,
    disappearance_granular_layer, follicular_horn_plug,
     scalp_involvement,
    knee_elbow_involvement, PNL_infiltrate,
     eosinophils_in_infiltrate,
    polygonal_papules,hyperkeratosis
]])

# Prediction button
if st.button("Predict Skin Disease"):
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        prediction = "Psoriasis: Yeh ek skin ki bimari hai jisme skin zyada tezi se banne lagti hai. Is wajah se skin par laal daag, sukhapan, aur chilna shuru ho jaata hai. Aksar khujli bhi hoti hai aur yeh ghutnon, kohniyon ya sir par zyada dekha jaata hai. Aap ki skin par laal daag ho kar chilti hai aur sukhapan hota hai. Kabhi kabhi khujli bhi hoti hai."
    elif prediction == 2:
        prediction = "Seborrheic Dermatitis (Seboric Dermatitis): Yeh aik aam bimari hai jo zyada tar sir ke balon mein ya chehre par hoti hai. Skin oily aur laal ho jaati hai, aur sukhay flakes (jaise ke dandruff) nazar aate hain. Sir ya chehre par sukhay chilke aur laali ho jaati hai. Jaise dandruff, lekin kabhi kabhi chehre par bhi."
    elif prediction == 3:
        prediction = "Lichen Planus (Liken Planus): Yeh skin aur kabhi kabhi mooh ke andar bhi hoti hai. Is mein chhoti chhoti baingani (purple) danay nikal aate hain, jo khujli karte hain. Jism par chhoti baingani danay nikalte hain jo khujli karte hain, kabhi kabhi mooh ke andar bhi ho jaata hai."
    elif prediction == 4:
        prediction = "Pityriasis Rosea (Pitiriyaasis Rosea): Yeh aik halka infection hota hai jisme pehle ek bada laal daag nikalta hai, phir us ke aas paas chhoti chhoti daag nikal aati hain. Yeh khud theek bhi ho jaata hai kuch hafton mein. Sabse pehle ek bada daag nikalta hai, phir chhoti chhoti daag us ke ird gird nikal aati hain. Thodi khujli bhi hoti hai, lekin yeh kuch hafton mein theek ho jaata hai."
    elif prediction == 5:
        prediction = "Chronic Dermatitis (Purani Sujan ya Chronic Dermatitis): Yeh ek purani skin problem hoti hai jisme skin bar bar laal, sukhay, aur khushk ho jaati hai. Kabhi kabhi phat jaati hai aur bar bar wapas aa jaata hai. Skin baar baar sukh jaati hai, laal ho jaati hai, aur phir phir waapas ho jaata hai. Jaise purani allergy ho."
    elif prediction == 6:
        prediction = "Pityriasis Rubra Pilaris (PRP): Yeh aik rare bimari hai jisme skin par sukhapan, laali aur chhoti chhoti daane nikalte hain. Aksar face, haath aur paon tak fail sakti hai. Rare bimari hai jisme skin sukh jaati hai, laal ho jaati hai aur chhoti chhoti danay nikalte hain. Poora jism bhi asar mein aa sakta hai."
    

    # Display the prediction result
    st.success(f"✅ Predicted Skin Disease Class: **{prediction}**")
    st.info("Please consult a dermatologist for a professional diagnosis.")
