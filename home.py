import streamlit as st

st.title('This is a title')
st.title('_Streamlit_ is :blue[cool] :sunglasses:')

st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')