import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import os

HISTORY_FILE = "/content/yoga_history.txt"

model = load_model(
    "/content/efficientnetb0_yoga_model.keras",
    custom_objects={'preprocess_input': preprocess_input}
)

class_names = [
    'Собака обличчям вниз', 'Поза низького випаду', 'Поза володаря риб', 'Поза метелика',
    'Поза журавля', 'Поза дитини', 'Поза корови', 'Поза орла', 'Поза плуга', 'Поза гірлянди',
    'Поза лотоса', 'Берізка', 'Поза колеса', 'Поза верблюда',
    'Поза стільця', 'Нахил вперед стоячи', 'Бокова планка'
]

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            return [line for line in lines if line.strip()]
    return []

def clear_plan():
      return ""

def save_to_history(entry):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(entry + '\n')

def analyze_video_with_plan(video_path, planned_text, add_to_history, history):
    if video_path is None:
        return "Завантажте відео для аналізу.", history, "\n".join(history) or ""

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = frame_count / fps if fps > 0 else 0

    frame_count = 0
    detected_poses = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            preds = model.predict(img, verbose=0)
            confidence = float(np.max(preds))
            if confidence > 0.5:
                pred_class = class_names[np.argmax(preds)]
                detected_poses.add(pred_class)

    cap.release()

    if not detected_poses:
        return "Асани не розпізнано або впевненість передбачення низька.", history, "\n".join(history) or ""

    planned_list = [p.strip() for p in planned_text.split('\n') if p.strip()]
    planned_set = set(planned_list)

    executed = planned_set.intersection(detected_poses)
    not_executed = planned_set - detected_poses

    result = ""
    result += f"Виконані з плану: {', '.join(sorted(executed)) or ''}\n"
    result += f"Не виконані з плану: {', '.join(sorted(not_executed)) or ''}\n"

    if add_to_history:
        date_str = datetime.now().strftime("%Y-%m-%d")
        history_entry = f"{date_str} | Виконані пози: {', '.join(sorted(executed)) or ''} | Тривалість: {total_duration:.0f} сек"
        save_to_history(history_entry)
        history = history + [history_entry]

    history_display = "\n".join(history) if history else ""

    return result, history, history_display

with gr.Blocks() as demo:
    # Зчитуємо історію з файлу при запуску
    initial_history = load_history()
    history_state = gr.State(initial_history)

    with gr.Row():
        video_input = gr.Video(
            label="Завантажити відео",
            sources=["upload"],
            format="mp4",
            interactive=True
        )

        with gr.Column():
            plan_text = gr.Textbox(
                label="План тренування",
                lines=6,
                interactive=False
            )

            plan_dropdown = gr.Dropdown(
                choices=class_names,
                label="Додати позу йоги до плану",
                allow_custom_value=False,
                multiselect=False,
                interactive=True
            )

            clear_btn = gr.Button("Очистити")
            clear_btn.click(
                fn=clear_plan,
                outputs=plan_text
            )

    add_to_history_checkbox = gr.Checkbox(
        label="Додати це тренування до 'Історії тренувань'",
        value=False
    )

    btn = gr.Button("Аналізувати відео та порівняти з планом")

    output = gr.Textbox(label="Результати порівняння", lines=8)

    history_output = gr.Textbox(
        label="Історія тренувань",
        value="\n".join(initial_history) or "",
        lines=10,
        interactive=False
    )

    def add_pose_to_plan(selected_pose, current_plan):
        if not selected_pose:
            return current_plan
        new_plan = current_plan.strip() + "\n" + selected_pose if current_plan else selected_pose
        return new_plan.strip()

    plan_dropdown.change(
        fn=add_pose_to_plan,
        inputs=[plan_dropdown, plan_text],
        outputs=plan_text
    )

    btn.click(
        fn=analyze_video_with_plan,
        inputs=[video_input, plan_text, add_to_history_checkbox, history_state],
        outputs=[output, history_state, history_output]
    )

demo.launch()
