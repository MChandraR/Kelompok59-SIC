import streamlit as st
import streamlit.components.v1 as components
import cv2
from PIL import Image

# Function to start video capture
def start_video():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stframe.image(frame, channels="BGR")

# Function to stop video capture
def stop_video():
    cv2.VideoCapture(0).release()

# Function to display team members
def display_team_member(image_path, name, role, instagram, linkedin, github):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, use_column_width=True)
    with col2:
        st.markdown(f"**{name}**")
        st.markdown(f"*{role}*")
        st.markdown(f"[Instagram]({instagram}) | [LinkedIn]({linkedin}) | [GitHub]({github})")

# Main app
def main():
    st.title("Integrity Vision")
    st.write("Proyek ini mengembangkan aplikasi website yang menggunakan teknologi kecerdasan buatan (AI) dan Internet of Things (IoT) untuk mendeteksi kecurangan selama ujian mahasiswa.")
    
    # Navigation menu
    menu = ["Home", "Tentang", "Mulai Ujian", "Hasil Ujian", "Tim"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.header("Home")
        st.video("path/to/video1.mp4")
        st.video("path/to/video2.mp4")
        st.video("path/to/video3.mp4")
        st.video("path/to/video4.mp4")
        st.write("Proyek ini mengembangkan aplikasi website yang menggunakan teknologi kecerdasan buatan (AI) dan Internet of Things (IoT) untuk mendeteksi kecurangan selama ujian mahasiswa.")
    
    elif choice == "Tentang":
        st.header("Tentang")
        st.write("**Batasan Masalah**")
        st.write("Ciri-ciri Kecurangan:")
        st.write("- Melihat teman kesamping dan kebelakang")
        st.write("- Membuka Handphone")
        st.image("path/to/image/melihat_teman.jpg", caption="Melihat Teman")
        st.image("path/to/image/melihathp.jpg", caption="Melihat Handphone")

    elif choice == "Mulai Ujian":
        st.header("Mulai Ujian")
        if st.button("Start Video"):
            start_video()
        if st.button("Stop Video"):
            stop_video()

    elif choice == "Hasil Ujian":
        st.header("Hasil Ujian")
        st.write("Coming Soon...")

    elif choice == "Tim":
        st.header("Our Team")
        display_team_member("path/to/image/1.png", "Akhiril Anwar Harahap", "Front-End Developer", "https://www.instagram.com/itsmevirgous/?igshid=ZGUzMzM3NWJiOQ%3D%3D", "https://www.linkedin.com/in/akhiril-anwar-harahap-8432851b9/", "https://github.com/akhirilanwarharahap")
        display_team_member("path/to/image/2.png", "Muhammad Chandra Ramadhan", "Front-End Developer", "#", "#", "#")
        display_team_member("path/to/image/3.png", "Luthfiyah Annisa", "UI Designer", "#", "#", "#")
        display_team_member("path/to/image/4.png", "Alika Dwi Yanti", "IOT Developer", "#", "#", "#")

    st.sidebar.markdown("### Support By:")
    st.sidebar.image("path/to/image/SIC.png", use_column_width=True)

if __name__ == "__main__":
    main()
