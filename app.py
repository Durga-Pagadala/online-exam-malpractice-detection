from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify , send_file
import cv2
import mediapipe as mp
import time
import sqlite3
import csv
# from flask import send_file
from io import StringIO

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- SETTINGS ----------------
PASS_MARK = 30
WARNING_LIMIT = 10

warning_count = 0
last_warning_time = 0

# ---------------- MEDIAPIPE SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True
)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            name TEXT,
            roll TEXT,
            email TEXT,
            score INTEGER,
            result TEXT,
            warnings INTEGER,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- CAMERA PROCESSING ----------------
def generate_frames():
    global warning_count, last_warning_time

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        height, width, _ = frame.shape

        # -------- NO FACE DETECTED --------
        if not results.multi_face_landmarks:
            if current_time - last_warning_time > 3:
                warning_count += 1
                last_warning_time = current_time

            cv2.putText(frame, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # -------- MULTIPLE FACES --------
            if len(results.multi_face_landmarks) > 1:
                if current_time - last_warning_time > 3:
                    warning_count += 1
                    last_warning_time = current_time

                cv2.putText(frame, "Multiple Faces Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            for face_landmarks in results.multi_face_landmarks:

                # -------- HEAD TURN DETECTION --------
                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * width)

                if nose_x < width * 0.35 or nose_x > width * 0.65:
                    if current_time - last_warning_time > 3:
                        warning_count += 1
                        last_warning_time = current_time

                    cv2.putText(frame, "Head Turn Detected", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # -------- IMPROVED EYE GAZE DETECTION --------
                try:
                    left_iris = face_landmarks.landmark[468]
                    left_eye_left = face_landmarks.landmark[33]
                    left_eye_right = face_landmarks.landmark[133]

                    iris_x = int(left_iris.x * width)
                    eye_left_x = int(left_eye_left.x * width)
                    eye_right_x = int(left_eye_right.x * width)

                    eye_width = eye_right_x - eye_left_x

                    if eye_width != 0:
                        gaze_ratio = (iris_x - eye_left_x) / eye_width

                        if gaze_ratio < 0.3 or gaze_ratio > 0.7:
                            if current_time - last_warning_time > 3:
                                warning_count += 1
                                last_warning_time = current_time

                            cv2.putText(frame, "Looking Away Detected", (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except:
                    pass

        # -------- DISPLAY WARNING COUNT --------
        cv2.putText(frame, f"Warnings: {warning_count}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ---------------- ROUTES ----------------

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['roll'] = request.form.get('roll')
        session['email'] = request.form.get('email')
        return redirect(url_for('exam'))

    return render_template('login.html')

@app.route('/exam')
def exam():
    if 'name' not in session:
        return redirect(url_for('home'))

    global warning_count
    warning_count = 0
    return render_template('exam.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tab_warning')
def tab_warning():
    global warning_count
    warning_count += 1
    return "OK"

@app.route('/check_warning')
def check_warning():
    return jsonify({"limit": warning_count >= WARNING_LIMIT})

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    global warning_count

    correct_answers = {
        "q1": "a",
        "q2": "b",
        "q3": "b",
        "q4": "a",
        "q5": "a"
    }

    score = 0

    for key in correct_answers:
        if request.form.get(key) == correct_answers[key]:
            score += 10

    result = "PASS" if score >= PASS_MARK else "FAIL"
    reason = "Auto Submitted (Cheating)" if warning_count >= WARNING_LIMIT else "Student Submitted"

    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?)",
              (session['name'],
               session['roll'],
               session['email'],
               score,
               result,
               warning_count,
               reason))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           name=session['name'],
                           roll=session['roll'],
                           email=session['email'],
                           score=score,
                           warnings=warning_count,
                           reason=reason,
                           passed=(score >= PASS_MARK))
    
    
@app.route('/admin')
def admin_dashboard():
    conn = sqlite3.connect("students.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    students = c.fetchall()
    conn.close()

    return render_template("admin.html", students=students)


@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    data = c.fetchall()
    conn.close()

    si = StringIO()
    writer = csv.writer(si)

    # Header row
    writer.writerow(["Name", "Roll", "Email", "Score", "Result", "Warnings", "Submission Type"])

    # Data rows
    for row in data:
        writer.writerow(row)

    output = si.getvalue()

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=exam_results.csv"}
    )

if __name__ == "__main__":
    app.run(debug=True)