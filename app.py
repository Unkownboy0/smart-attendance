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
import pyttsx3
import threading
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.ttk as ttk
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import qrcode
from cryptography.fernet import Fernet
import itertools
import numpy as np

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

        tk.Button(self.main_window, text="Toggle Theme", command=self.toggle_theme).place(x=800, y=20)

    def load_or_create_attendance_csv(self):
        columns = ["Name", "Time", "Date", "Image"]
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
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
        known_encodings = []
        known_names = []
        for file in os.listdir(self.db_dir):
            if file.endswith('.pickle'):
                name = file[:-7]
                with open(os.path.join(self.db_dir, file), 'rb') as f:
                    encoding = pickle.load(f)
                    known_encodings.append(encoding)
                    known_names.append(name)
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "unknown_person"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
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

        if not hasattr(self, "_recognition_thread_running"):
            self._recognition_thread_running = False

        if not self._recognition_thread_running:
            self._recognition_thread_running = True

            def recognition_task():
                try:
                    self.handle_group_attendance(frame)
                finally:
                    self._recognition_thread_running = False

            threading.Thread(target=recognition_task, daemon=True).start()

        self._label.after(20, self.process_webcam)

    def handle_group_attendance(self, frame):
        names, locations = self.recognize_faces_in_frame(frame)
        for name in names:
            if name not in ['unknown_person', 'no_persons_found']:
                if name not in self.attendance_log:
                    self.mark_attendance(name)

    def mark_attendance(self, name):
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        img_path = self.save_attendance_image(name)
        if name not in self.attendance_log:
            self.attendance_log[name] = now
            self.df.loc[len(self.df)] = [name, time_str, date_str, img_path]
            self.df.to_csv(ATTENDANCE_FILE, index=False)
            print(f"[+] Marked attendance for {name} at {time_str}")

            self.speak_welcome(name)

            user_email_path = os.path.join(self.db_dir, f"{name}.email")
            user_email = None
            if os.path.exists(user_email_path):
                with open(user_email_path, "r") as f:
                    user_email = f.read().strip()
            subject = "present"
            body = f"{name} present.\nTime: {time_str}"
            # Always send to admin
            self.send_email(subject, body, "unkownnew19@gmail.com", image_path=img_path)
            # Send to user if available
            if user_email:
                self.send_email(subject, body, user_email, image_path=img_path)

    def speak_welcome(self, name):
        def speak():
            try:
                engine = pyttsx3.init()
                engine.say(f"Welcome {name}")
                engine.runAndWait()
            except Exception as e:
                print(f"Could not speak welcome: {e}")
        threading.Thread(target=speak, daemon=True).start()

    def mark_absent(self, absent_list):
        absent_str = "\n".join(absent_list)
        self.send_email("absenties", f"Absenties list:\n{absent_str}", "unkownnew19@gmail.com")

    def login(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', f'Welcome, {name}.')
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
            img_path = self.save_attendance_image(name)
            with open(self.log_path, 'a') as f:
                f.write(f"{name},{now},in\n")
            self.mark_attendance(name)
            # Send email to user and admin
            user_email_path = os.path.join(self.db_dir, f"{name}.email")
            user_email = None
            if os.path.exists(user_email_path):
                with open(user_email_path, "r") as f:
                    user_email = f.read().strip()
            subject = "Login Notification"
            body = f"{name} logged in at {time_str}."
            self.send_email(subject, body, "unkownnew19@gmail.com", image_path=img_path)
            if user_email:
                self.send_email(subject, body, user_email, image_path=img_path)

    def logout(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Hasta la vista !', f'See you again {name}.')
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
            img_path = self.save_attendance_image(name)
            with open(self.log_path, 'a') as f:
                f.write(f"{name},{now},out\n")
            self.speak_goodbye(name)
            self.mark_leave(name)
            # Send email to user and admin
            user_email_path = os.path.join(self.db_dir, f"{name}.email")
            user_email = None
            if os.path.exists(user_email_path):
                with open(user_email_path, "r") as f:
                    user_email = f.read().strip()
            subject = "Logout Notification"
            body = f"{name} logged out at {time_str}."
            self.send_email(subject, body, "unkownnew19@gmail.com", image_path=img_path)
            if user_email:
                self.send_email(subject, body, user_email, image_path=img_path)

    def speak_goodbye(self, name):
        def speak():
            try:
                engine = pyttsx3.init()
                engine.say(f"See you again {name}")
                engine.runAndWait()
            except Exception as e:
                print(f"Could not speak goodbye: {e}")
        threading.Thread(target=speak, daemon=True).start()

    def mark_leave(self, name):
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        print(f"[+] Marked leave for {name} at {time_str}")

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
            with open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb') as file:
                pickle.dump(embedding, file)
            with open(os.path.join(self.db_dir, '{}.email'.format(name)), 'w') as file:
                file.write(email)
            util.msg_box('Success!', 'User was registered successfully !')
        else:
            util.msg_box('Error!', 'No face detected. Please try again.')
        self.register_new_user_window.destroy()

    def edit_user(self):
        edit_window = tk.Toplevel(self.main_window)
        edit_window.geometry("400x350")
        edit_window.configure(bg="#e3f2fd")
        edit_window.title("Edit/Delete User")

        user_list = [f[:-7] for f in os.listdir(self.db_dir) if f.endswith('.pickle')]
        user_var = tk.StringVar(edit_window)
        user_var.set(user_list[0] if user_list else "")

        tk.Label(edit_window, text="Select user:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 12, "bold")).pack(pady=10)
        user_menu = tk.OptionMenu(edit_window, user_var, *user_list)
        user_menu.pack(pady=10)

        tk.Label(edit_window, text="New name:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 12)).pack()
        new_name_entry = tk.Entry(edit_window)
        new_name_entry.pack(pady=5)

        tk.Label(edit_window, text="New email:", bg="#e3f2fd", fg="#0d47a1", font=("Segoe UI", 12)).pack()
        new_email_entry = tk.Entry(edit_window)
        new_email_entry.pack(pady=5)

        def delete_user():
            user = user_var.get()
            file_path = os.path.join(self.db_dir, f"{user}.pickle")
            email_path = os.path.join(self.db_dir, f"{user}.email")
            if os.path.exists(file_path):
                os.remove(file_path)
                if os.path.exists(email_path):
                    os.remove(email_path)
                tk.messagebox.showinfo("Success", f"User '{user}' deleted.")
                edit_window.destroy()
            else:
                tk.messagebox.showerror("Error", "User file not found.")

        def update_user():
            user = user_var.get()
            new_name = new_name_entry.get().strip().replace('\n', '').replace('\r', '')
            new_email = new_email_entry.get().strip()
            invalid_chars = r'\/:*?"<>|'
            for ch in invalid_chars:
                new_name = new_name.replace(ch, '_')
            old_pickle = os.path.join(self.db_dir, f"{user}.pickle")
            old_email = os.path.join(self.db_dir, f"{user}.email")
            new_pickle = os.path.join(self.db_dir, f"{new_name}.pickle")
            new_email_file = os.path.join(self.db_dir, f"{new_name}.email")
            if os.path.exists(old_pickle):
                os.rename(old_pickle, new_pickle)
                if os.path.exists(old_email):
                    os.rename(old_email, new_email_file)
                with open(new_email_file, "w") as f:
                    f.write(new_email)
                tk.messagebox.showinfo("Success", f"User '{user}' updated.")
                edit_window.destroy()
            else:
                tk.messagebox.showerror("Error", "User file not found.")

        tk.Button(edit_window, text="Delete User", bg="#F44336", fg="white", command=delete_user).pack(pady=10)
        tk.Button(edit_window, text="Update User", bg="#4CAF50", fg="white", command=update_user).pack(pady=10)

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
            os.makedirs(backup_dir)
        while True:
            try:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = os.path.join(backup_dir, f'attendance_{timestamp}.csv')
                self.df.to_csv(backup_file, index=False)
                print(f"Backup saved: {backup_file}")
            except Exception as e:
                print(f"Backup failed: {e}")
            time.sleep(3600)

    def show_qr_backup(self, name):
        qr = qrcode.make(f"Attendance:{name}:{datetime.datetime.now()}")
        qr.show()

    def toggle_theme(self):
        if self.main_window['bg'] == "#e3f2fd":
            self.main_window.configure(bg="#222831")
        else:
            self.main_window.configure(bg="#e3f2fd")

    def fingerprint_authenticate(self):
        print("Fingerprint authentication not implemented.")
        return False

    def show_fingerprint_animation(self, on_success=None, on_fail=None):
        anim_window = tk.Toplevel(self.main_window)
        anim_window.title("Fingerprint Authentication")
        anim_window.geometry("300x400")
        anim_window.configure(bg="#e3f2fd")
        anim_window.resizable(False, False)

        gif_path = "fingerprint.gif"
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frames.append(ImageTk.PhotoImage(gif.copy()))
                gif.seek(len(frames))
        except EOFError:
            pass

        label = tk.Label(anim_window, bg="#e3f2fd")
        label.pack(expand=True)

        def animate(counter=0):
            label.config(image=frames[counter])
            counter = (counter + 1) % len(frames)
            anim_window.after(60, animate, counter)

        animate()

        def finish_scan():
            anim_window.destroy()
            if self.fingerprint_authenticate():
                if on_success:
                    on_success()
            else:
                if on_fail:
                    on_fail()

        anim_window.after(2000, finish_scan)
        self.show_fingerprint_animation(on_success=lambda: util.msg_box("Success", "Fingerprint recognized!"))

    def save_attendance_image(self, name):
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(IMAGE_DIR, f"{name}_{now}.jpg")
        cv2.imwrite(img_path, self.most_recent_capture_arr)
        return img_path

    def send_email(self, subject, body, to_email, image_path=None):
        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "workerj139@gmail.com"
            sender_password = "hhip ilor gkfj huyc"

            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = to_email

            msg.attach(MIMEText(body, "plain"))

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
                msg.attach(part)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, to_email, msg.as_string())
            print(f"Email sent to {to_email}")
        except Exception as e:
            print(f"Email send failed: {e}")

if __name__ == "__main__":
    app = App()
    app.main_window.mainloop()
