import os
import datetime
import pickle
import pandas as pd
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
from playsound import playsound
import util
from mytest import test
import pyttsx3
import threading
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.ttk as ttk
import time
import smtplib
from email.mime.text import MIMEText
import concurrent.futures
import mediapipe as mp
from cryptography.fernet import Fernet
import qrcode

ATTENDANCE_FILE = os.path.abspath("attendance.csv")
WELCOME_SOUND = os.path.abspath("welcome.mp3")
IMAGE_DIR = os.path.abspath("./attendance_images")

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Face Recognition Attendance")
        self.main_window.geometry("900x600")
        self.main_window.configure(bg="#e3f2fd")

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=30, y=30, width=500, height=400)
        self.add_webcam(self.webcam_label)

        # Button frame for neat layout
        button_frame = tk.Frame(self.main_window, bg="#e3f2fd")
        button_frame.place(x=600, y=80, width=250, height=350)

        self.login_button = util.get_button(button_frame, 'Login', '#4CAF50', self.login)
        self.login_button.pack(pady=10, fill="x")

        self.logout_button = util.get_button(button_frame, 'Logout', '#F44336', self.logout)
        self.logout_button.pack(pady=10, fill="x")

        self.register_button = util.get_button(button_frame, 'Register User', '#2196F3', self.register_new_user)
        self.register_button.pack(pady=10, fill="x")

        self.edit_user_button = util.get_button(button_frame, 'Edit/Delete User', '#FF9800', self.edit_user)
        self.edit_user_button.pack(pady=10, fill="x")

        self.report_button = util.get_button(button_frame, 'Reports', '#9C27B0', self.show_reports)
        self.report_button.pack(pady=10, fill="x")

        self.db_dir = os.path.abspath('./db')
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        if not os.path.exists(IMAGE_DIR):
            os.mkdir(IMAGE_DIR)

        self.log_path = os.path.abspath('./log.txt')
        self.df = self.load_or_create_attendance_csv()
        self.attendance_log = {}

        self.setup_theme()
        self.sync_attendance()
        threading.Thread(target=self.auto_backup, daemon=True).start()

        # Add a button to toggle theme
        tk.Button(self.main_window, text="Toggle Theme", command=self.toggle_theme).place(x=800, y=20)

    def load_or_create_attendance_csv(self):
        columns = ["Name", "Time", "Date", "Image"]
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                # Ensure columns match
                for col in columns:
                    if col not in df.columns:
                        df[col] = ""
                df = df[columns]
                return df
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=columns)
        else:
            return pd.DataFrame(columns=columns)

    def add_webcam(self, label):
        try:
            if 'cap' not in self.__dict__:
                self.cap = cv2.VideoCapture(0)
            self._label = label
            self.process_webcam()
        except Exception as e:
            print(f"Webcam error: {e}")

    def recognize_faces_in_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        for encoding in face_encodings:
            name = util.recognize_encoding(encoding, self.db_dir)
            names.append(name)
        return names, face_locations

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture webcam frame.")
            return
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        # Run recognition in a thread
        threading.Thread(target=self.handle_group_attendance, args=(frame,), daemon=True).start()
        self._label.after(20, self.process_webcam)

    def is_live_face(self, frame):
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                # Add blink/head-turn logic here
                return True
        return False

    def handle_group_attendance(self, frame):
        if not self.is_live_face(frame):
            print("Spoof detected!")
            return
        names, locations = self.recognize_faces_in_frame(frame)
        for name in names:
            if name not in ['unknown_person', 'no_persons_found']:
                self.mark_attendance(name)

    def save_attendance_image(self, name):
        now = datetime.datetime.now()
        img_filename = f"{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        cv2.imwrite(img_path, self.most_recent_capture_arr)
        return img_path

    def send_email(self, subject, body, to_addr="unkownnew19@gmail.com"):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = "workerj139@gmail.com"
        msg['To'] = to_addr
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login("workerj139@gmail.com", "zxpu szzi truj ovbh")
                server.send_message(msg)
        except Exception as e:
            print(f"Email send failed: {e}")

    def get_location(self):
        try:
            response = requests.get('https://ipinfo.io/json')
            data = response.json()
            return data.get('loc', '')
        except Exception:
            return ''

    def mark_attendance(self, name):
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        img_path = self.save_attendance_image(name)
        location = self.get_location()
        if name not in self.attendance_log:
            self.attendance_log[name] = now
            self.df.loc[len(self.df)] = [name, time_str, date_str, img_path]
            try:
                self.df.to_csv(ATTENDANCE_FILE, index=False)
            except PermissionError:
                print(f"Permission denied: Could not write to {ATTENDANCE_FILE}. Please close the file if it's open in another program.")
            print(f"[+] Marked attendance for {name} at {time_str}")

            # Run TTS in a separate thread
            def speak_welcome():
                try:
                    engine = pyttsx3.init()
                    engine.say(f"Welcome {name}")
                    engine.runAndWait()
                except Exception as e:
                    print(f"Could not speak welcome: {e}")

            threading.Thread(target=speak_welcome, daemon=True).start()

            try:
                if os.path.exists(WELCOME_SOUND):
                    playsound(WELCOME_SOUND)
                else:
                    print(f"Sound file not found: {WELCOME_SOUND}")
            except Exception as e:
                print(f"Could not play sound: {e}")
            # Send present email
            self.send_email("presents", f"{name} is present.", "unkownnew19@gmail.com")

    def mark_absent(self, absent_list):
        # absent_list: list of names
        absent_str = "\n".join(absent_list)
        self.send_email("absenties", f"Absenties list:\n{absent_str}", "unkownnew19@gmail.com")

    def login(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='./models',
            device_id=0
        )
        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', f'Welcome, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                self.mark_attendance(name)
        else:
            util.msg_box('Hey, you are a spoofer!', 'Photo or spoof detected. Login stopped.')

    def logout(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='./models',
            device_id=0
        )
        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', f'See you again {name}.')
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                try:
                    engine = pyttsx3.init()
                    engine.say(f"See you again {name}")
                    engine.runAndWait()
                except Exception as e:
                    print(f"Could not speak goodbye: {e}")
                # Also mark leave here
                self.mark_leave(name)
        else:
            util.msg_box('Hey, you are a spoofer!', 'Photo or spoof detected. Logout stopped.')

    def mark_leave(self, name=None):
        if name is None:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            return
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        # Save leave image
        img_path = self.save_attendance_image(name)
        leave_columns = ["Name", "Time", "Date", "Image"]
        if not hasattr(self, "leave_df"):
            self.leave_df = pd.DataFrame(columns=leave_columns)
        if os.path.exists("leave.csv"):
            try:
                leave_df = pd.read_csv("leave.csv")
                for col in leave_columns:
                    if col not in leave_df.columns:
                        leave_df[col] = ""
                leave_df = leave_df[leave_columns]
                self.leave_df = leave_df
            except pd.errors.EmptyDataError:
                self.leave_df = pd.DataFrame(columns=leave_columns)
        self.leave_df.loc[len(self.leave_df)] = [name, time_str, date_str, img_path]
        try:
            self.leave_df.to_csv("leave.csv", index=False)
        except PermissionError:
            print(f"Permission denied: Could not write to leave.csv. Please close the file if it's open in another program.")
        util.msg_box('Leave Marked', f'Leave marked for {name} at {time_str} on {date_str}.')

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("600x400")
        self.register_new_user_window.configure(bg="#e3f2fd")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', '#4CAF50', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=350, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', '#F44336', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=350, y=350)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=30, y=30, width=250, height=200)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=350, y=100)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Input username:')
        self.text_label_register_new_user.place(x=350, y=60)

        # Email entry
        tk.Label(self.register_new_user_window, text="Email ID:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 14)).place(x=350, y=160)
        self.email_entry = tk.Entry(self.register_new_user_window, font=("Segoe UI", 14), bg="#e3f2fd", fg="#0d47a1", width=22)
        self.email_entry.place(x=350, y=190)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip().replace('\n', '').replace('\r', '')
        email = self.email_entry.get().strip()
        invalid_chars = r'\/:*?"<>|'
        for ch in invalid_chars:
            name = name.replace(ch, '_')
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if embeddings:
            embedding = embeddings[0]
            # Save embedding
            with open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb') as file:
                pickle.dump(embedding, file)
            # Save email to a file
            with open(os.path.join(self.db_dir, '{}.email'.format(name)), 'w') as file:
                file.write(email)
            util.msg_box('Success!', 'User was registered successfully !')
        else:
            util.msg_box('Error!', 'No face detected. Please try again.')
        self.register_new_user_window.destroy()

    def edit_user(self):
        edit_window = tk.Toplevel(self.main_window)
        edit_window.geometry("400x300")
        edit_window.configure(bg="#e3f2fd")
        edit_window.title("Edit/Delete User")

        user_list = [f[:-7] for f in os.listdir(self.db_dir) if f.endswith('.pickle')]
        user_var = tk.StringVar(edit_window)
        user_var.set(user_list[0] if user_list else "")

        tk.Label(edit_window, text="Select user:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 12, "bold")).pack(pady=10)
        user_menu = tk.OptionMenu(edit_window, user_var, *user_list)
        user_menu.pack(pady=10)

        def delete_user():
            user = user_var.get()
            file_path = os.path.join(self.db_dir, f"{user}.pickle")
            if os.path.exists(file_path):
                os.remove(file_path)
                tk.messagebox.showinfo("Success", f"User '{user}' deleted.")
                edit_window.destroy()
            else:
                tk.messagebox.showerror("Error", "User file not found.")

        def rename_user():
            user = user_var.get()
            new_name = new_name_entry.get().strip().replace('\n', '').replace('\r', '')
            invalid_chars = r'\/:*?"<>|'
            for ch in invalid_chars:
                new_name = new_name.replace(ch, '_')
            old_path = os.path.join(self.db_dir, f"{user}.pickle")
            new_path = os.path.join(self.db_dir, f"{new_name}.pickle")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                tk.messagebox.showinfo("Success", f"User '{user}' renamed to '{new_name}'.")
                edit_window.destroy()
            else:
                tk.messagebox.showerror("Error", "User file not found.")

        tk.Button(edit_window, text="Delete User", bg="#F44336", fg="white", command=delete_user).pack(pady=10)

        tk.Label(edit_window, text="New name for user:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 12)).pack(pady=10)
        new_name_entry = tk.Entry(edit_window)
        new_name_entry.pack(pady=10)
        tk.Button(edit_window, text="Rename User", bg="#4CAF50", fg="white", command=rename_user).pack(pady=10)

    def sync_attendance(self):
        def sync_thread():
            while True:
                try:
                    requests.get("https://www.google.com", timeout=5)
                    print("Network available, syncing...")
                    break
                except requests.ConnectionError:
                    print("No network, will retry...")
                    time.sleep(60)
        threading.Thread(target=sync_thread, daemon=True).start()

    def show_reports(self):
        report_window = tk.Toplevel(self.main_window)
        report_window.geometry("800x600")
        report_window.title("Attendance Reports")
        report_window.configure(bg="#e3f2fd")

        df = self.df.copy()
        # Add filter widgets here (subject, department, time slot)
        # ...existing code for graphs...
        # Add more graphs for trends, frequent absentees, etc.

        df['Late'] = df['Time'] > '09:00:00'
        late_count = df[df['Late']].groupby('Name').size()
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        monthly_summary = df.groupby(['Name', 'Month']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(4,2))
        late_count.plot(kind='bar', ax=ax, color="#4CAF50")
        ax.set_title("Late Comers")
        ax.set_ylabel("Count")
        canvas = FigureCanvasTkAgg(fig, master=report_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        def export_pdf():
            fig.savefig("attendance_report.pdf")
            tk.messagebox.showinfo("Export", "PDF exported!")

        def export_excel():
            df.to_excel("attendance_report.xlsx", index=False)
            tk.messagebox.showinfo("Export", "Excel exported!")

        ttk.Button(report_window, text="Export PDF", command=export_pdf).pack(pady=10)
        ttk.Button(report_window, text="Export Excel", command=export_excel).pack(pady=10)

    def setup_theme(self):
        style = ttk.Style(self.main_window)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 12))
        style.configure('TButton', padding=6, relief="flat", background="#4CAF50", foreground="white")
        self.main_window.configure(bg="#e3f2fd")

    def back_to_menu(self):
        pass

    def turn_on_camera(self):
        pass

    def get_crypto_key(self):
        key_path = os.path.join(self.db_dir, 'key.key')
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
        else:
            with open(key_path, 'rb') as f:
                key = f.read()
        return Fernet(key)

    def save_encrypted_embedding(self, name, embedding):
        fernet = self.get_crypto_key()
        data = pickle.dumps(embedding)
        encrypted = fernet.encrypt(data)
        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
            file.write(encrypted)

    def load_encrypted_embedding(self, name):
        fernet = self.get_crypto_key()
        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'rb') as file:
            encrypted = file.read()
        data = fernet.decrypt(encrypted)
        return pickle.loads(data)

    def auto_backup(self):
        backup_dir = os.path.abspath('./backup')
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        while True:
            try:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = os.path.join(backup_dir, f'attendance_{timestamp}.csv')
                self.df.to_csv(backup_file, index=False)
                print(f"Backup saved: {backup_file}")
            except Exception as e:
                print(f"Backup failed: {e}")
            time.sleep(3600)  # every hour

    def predict_low_attendance(self):
        df = self.df.copy()
        attendance_counts = df.groupby('Name').size()
        threshold = 0.75 * df['Date'].nunique()  # e.g., 75% attendance
        low_attendees = attendance_counts[attendance_counts < threshold]
        return low_attendees.index.tolist()

    def show_qr_backup(self, name):
        qr = qrcode.make(f"Attendance:{name}:{datetime.datetime.now()}")
        qr.show()

    def toggle_theme(self):
        if self.main_window['bg'] == "#e3f2fd":
            self.main_window.configure(bg="#222831")
            # Update widget colors for dark mode
        else:
            self.main_window.configure(bg="#e3f2fd")
            # Update widget colors for light mode

if __name__ == "__main__":
    app = App()
    app.main_window.mainloop()
