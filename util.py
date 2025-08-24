import os
import pickle

import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
        window,
        text=text,
        activebackground="#1976d2",
        activeforeground="#fff",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Segoe UI', 18, 'bold'),
        relief="flat",
        borderwidth=0,
        highlightthickness=0,
        cursor="hand2"
    )
    button.configure(
        highlightbackground="#1565c0",
        highlightcolor="#1565c0"
    )
    return button


def get_img_label(window):
    label = tk.Label(window, bg="#1976d2", bd=2, relief="groove")
    label.place(x=10, y=0, width=700, height=500)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text, bg="#2196f3", fg="white")
    label.config(font=("Segoe UI", 20, "bold"), justify="left", anchor="w", padx=10)
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15,
                       font=("Segoe UI", 24, "bold"),
                       bg="#e3f2fd",
                       fg="#0d47a1",
                       bd=2,
                       relief="groove")
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted([f for f in os.listdir(db_path) if f.endswith('.pickle')])  # Only .pickle files

    match = False
    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])
        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)
        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    if match:
        return db_dir[j - 1][:-7]
    else:
        return 'unknown_person'

