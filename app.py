import streamlit as st
import util
from PIL import Image

st.set_page_config(page_title='Sports Celebrity Image Classification', layout = 'centered', initial_sidebar_state = 'expanded')

util.local_css("style.css")
util.load_saved_artifacts()

paths = ['lionel.jpg', 'federer.jpg', 'virat1.jpg', 'sharapova.jpg', 'virat2.jpg', 'serena.jpg']
cols = st.columns(3)
for index, path in enumerate(paths):
  with cols[(index%3)]:
    img = Image.open('test_images/'+path)
    st.image(img, caption=paths[index].split('.')[0].capitalize())

uploaded_file = st.file_uploader(label="Select Image",type=['png', 'jpg'], label_visibility="hidden")

if uploaded_file is not None:
  result = util.classify_image(uploaded_file)
  if result != 0:
    player_name = result[0]["class"]
    probability = result[0]["class_probability"][result[0]["class_dictionary"][player_name]]
    st.markdown("<div class='flex'><h3>Player Name: "+player_name+"</h3> <h3>Probability: "+str(probability)+"</h3></div>",unsafe_allow_html=True)
  else:
    st.markdown("<h2 class='center'>Sorry, The image can't be identified</h2>",unsafe_allow_html=True)
