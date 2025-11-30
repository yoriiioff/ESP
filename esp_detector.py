import cv2
import os
import sys
import tempfile
import subprocess
from ultralytics import YOLO
import torch

# GUI imports
try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    CUSTOM_TKINTER_AVAILABLE = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    CUSTOM_TKINTER_AVAILABLE = False

def detect_and_draw_esp(frame, model):
    """
    Обнаруживает объекты и рисует ESP-подобные элементы
    """
    results = model(frame)

    # Все классы YOLOv8 COCO dataset для обнаружения (автоматически)
    # Создаем словарь с человекопонятными названиями для всех классов
    target_classes = {}
    for class_id, class_name in model.names.items():
        # Для всех классов используем их оригинальные названия, но с заглавной буквы
        target_classes[class_name] = class_name.replace('_', ' ').title()

    detected_classes = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Получаем координаты и класс
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                class_name = result.names[cls]
                detected_classes.append(f"{class_name} ({conf:.2f})")

                # Обнаруживаем все классы из YOLOv8 COCO dataset
                # Используем стандартный порог уверенности 0.5 для всех объектов
                confidence_threshold = 0.5
                if class_name in target_classes and conf > confidence_threshold:
                    # Рисуем красный прямоугольник
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Добавляем подпись
                    label = target_classes[class_name]
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame, detected_classes

def process_video(video_path, output_path, progress_callback=None, log_callback=None):
    """
    Обрабатывает видео и сохраняет результат с сохранением аудио
    """
    # Загружаем модель YOLOv8
    if log_callback:
        log_callback("Загрузка модели YOLOv8...")
    else:
        print("Загрузка модели YOLOv8...")
    model = YOLO('yolov8n.pt')  # Используем nano версию для скорости

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"Ошибка открытия видео: {video_path}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return

    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создаем временный файл для видео без звука
    temp_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    # Создаем VideoWriter для временного файла
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if log_callback:
        log_callback(f"Обработка видео: {total_frames} кадров")
    else:
        print(f"Обработка видео: {total_frames} кадров")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр
        processed_frame, detected_classes = detect_and_draw_esp(frame, model)

        # Записываем обработанный кадр
        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Показываем прогресс каждые 30 кадров
            progress = (frame_count / total_frames) * 100
            if progress_callback:
                progress_callback(progress)
            if log_callback:
                log_callback(f"Прогресс: {progress:.1f}%")
                if detected_classes:
                    log_callback(f"Обнаруженные объекты: {', '.join(detected_classes[:5])}")
            else:
                print(f"Прогресс: {progress:.1f}%")
                if detected_classes:
                    print(f"Обнаруженные объекты: {', '.join(detected_classes[:5])}")

    # Освобождаем ресурсы OpenCV
    cap.release()
    out.release()

    # Теперь объединяем видео с аудио с помощью ffmpeg
    try:
        if log_callback:
            log_callback("Объединение видео и аудио с помощью ffmpeg...")
        else:
            print("Объединение видео и аудио с помощью ffmpeg...")

        ffmpeg_path = r"N:\ai maked ai\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

        if os.path.exists(ffmpeg_path):
            # Используем ffmpeg для объединения
            cmd = [
                ffmpeg_path, '-i', temp_video_path, '-i', video_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', output_path, '-y'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode == 0:
                success_msg = "Аудио успешно объединено с видео!"
                if log_callback:
                    log_callback(success_msg)
                else:
                    print(success_msg)
            else:
                error_msg = f"Ошибка ffmpeg: {result.stderr}"
                if log_callback:
                    log_callback(error_msg)
                else:
                    print(error_msg)
                # Fallback: копируем видео без звука
                import shutil
                shutil.copy2(temp_video_path, output_path)
        else:
            warning_msg = "FFmpeg не найден, видео сохранено без аудио"
            if log_callback:
                log_callback(warning_msg)
            else:
                print(warning_msg)
            import shutil
            shutil.copy2(temp_video_path, output_path)

    except Exception as e:
        error_msg = f"Ошибка при объединении аудио: {e}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        # Fallback: копируем видео без звука
        import shutil
        shutil.copy2(temp_video_path, output_path)
        fallback_msg = "Видео сохранено без аудио"
        if log_callback:
            log_callback(fallback_msg)
        else:
            print(fallback_msg)

    # Удаляем временный файл
    try:
        os.unlink(temp_video_path)
    except:
        pass

    success_msg = f"Обработка завершена! Результат сохранен в: {output_path}"
    if log_callback:
        log_callback(success_msg)
    else:
        print(success_msg)

class ESPDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP для видео - Обнаружение объектов в видео")
        self.root.geometry("700x500")
        self.root.resizable(True, True)

        # Переменные
        self.input_file = ""
        self.output_file = ""
        self.processing = False

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        if CUSTOM_TKINTER_AVAILABLE:
            # CustomTkinter стиль
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

            # Главный фрейм
            main_frame = ctk.CTkFrame(self.root)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Заголовок
            title_label = ctk.CTkLabel(main_frame, text="ESP для видео",
                                     font=ctk.CTkFont(size=16, weight="bold"))
            title_label.pack(pady=(10, 20))

            # Выбор файла
            file_frame = ctk.CTkFrame(main_frame)
            file_frame.pack(fill="x", padx=20, pady=(0, 20))

            ctk.CTkLabel(file_frame, text="Видео файл:").pack(side="left", padx=(10, 5))

            self.file_entry = ctk.CTkEntry(file_frame, width=400)
            self.file_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)

            browse_btn = ctk.CTkButton(file_frame, text="Обзор", width=80,
                                     command=self.browse_file)
            browse_btn.pack(side="right", padx=(0, 10))

            # Кнопки управления
            button_frame = ctk.CTkFrame(main_frame)
            button_frame.pack(fill="x", padx=20, pady=(0, 20))

            self.process_btn = ctk.CTkButton(button_frame, text="Обработать видео",
                                           command=self.start_processing)
            self.process_btn.pack(side="left", padx=(10, 10))

            self.open_output_btn = ctk.CTkButton(button_frame, text="Открыть результат",
                                               command=self.open_output, state="disabled")
            self.open_output_btn.pack(side="left", padx=(0, 10))

            # Прогресс бар
            progress_frame = ctk.CTkFrame(main_frame)
            progress_frame.pack(fill="x", padx=20, pady=(0, 20))

            ctk.CTkLabel(progress_frame, text="Прогресс:").pack(anchor="w", padx=10, pady=(10, 5))

            self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
            self.progress_bar.pack(padx=10, pady=(0, 10))
            self.progress_bar.set(0)

            # Лог область
            log_frame = ctk.CTkFrame(main_frame)
            log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

            ctk.CTkLabel(log_frame, text="Лог обработки:").pack(anchor="w", padx=10, pady=(10, 5))

            self.log_textbox = ctk.CTkTextbox(log_frame, wrap="word")
            self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            # Статус
            self.status_label = ctk.CTkLabel(main_frame, text="Готов к работе")
            self.status_label.pack(pady=(0, 10))

        else:
            # Обычный tkinter
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill="both", expand=True)

            # Заголовок
            title_label = ttk.Label(main_frame, text="ESP для видео",
                                   font=("Arial", 16, "bold"))
            title_label.pack(pady=(0, 20))

            # Выбор файла
            file_frame = ttk.Frame(main_frame)
            file_frame.pack(fill="x", pady=(0, 20))

            ttk.Label(file_frame, text="Видео файл:").pack(side="left", padx=(0, 5))

            self.file_entry = ttk.Entry(file_frame, width=50)
            self.file_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)

            browse_btn = ttk.Button(file_frame, text="Обзор", command=self.browse_file)
            browse_btn.pack(side="right")

            # Кнопки управления
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill="x", pady=(0, 20))

            self.process_btn = ttk.Button(button_frame, text="Обработать видео",
                                        command=self.start_processing)
            self.process_btn.pack(side="left", padx=5)

            self.open_output_btn = ttk.Button(button_frame, text="Открыть результат",
                                            command=self.open_output, state="disabled")
            self.open_output_btn.pack(side="left", padx=5)

            # Прогресс бар
            ttk.Label(main_frame, text="Прогресс:").pack(anchor="w", pady=(0, 5))

            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                               maximum=100, length=400)
            self.progress_bar.pack(fill="x", pady=(0, 20))

            # Лог область
            ttk.Label(main_frame, text="Лог обработки:").pack(anchor="w", pady=(0, 5))

            log_frame = ttk.Frame(main_frame)
            log_frame.pack(fill="both", expand=True)

            self.log_text = tk.Text(log_frame, height=10, wrap="word")
            scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
            self.log_text.configure(yscrollcommand=scrollbar.set)

            self.log_text.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Статус
            self.status_var = tk.StringVar()
            self.status_var.set("Готов к работе")
            status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                                 relief="sunken", anchor="w")
            status_bar.pack(fill="x", pady=(20, 0))

    def browse_file(self):
        """Выбор входного файла"""
        filetypes = [
            ('Видео файлы', '*.mp4 *.avi *.mov *.mkv *.wmv'),
            ('Все файлы', '*.*')
        ]

        filename = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=filetypes
        )

        if filename:
            self.input_file = filename
            self.file_entry.delete(0, tk.END) if not CUSTOM_TKINTER_AVAILABLE else self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)

            # Определяем выходной файл
            video_dir = os.path.dirname(filename) or "."
            self.output_file = os.path.join(video_dir, "out.mp4")

            self.log_message(f"Выбран файл: {filename}")
            self.log_message(f"Результат будет сохранен: {self.output_file}")

    def log_message(self, message):
        """Добавление сообщения в лог"""
        if CUSTOM_TKINTER_AVAILABLE:
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.see("end")
        else:
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
        self.root.update_idletasks()

    def start_processing(self):
        """Запуск обработки видео"""
        if not self.input_file:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите видео файл!")
            return

        if not os.path.exists(self.input_file):
            messagebox.showerror("Ошибка", f"Файл не найден: {self.input_file}")
            return

        if self.processing:
            return

        # Запуск обработки в отдельном потоке
        import threading
        self.processing = True
        self.process_btn.configure(state="disabled") if CUSTOM_TKINTER_AVAILABLE else self.process_btn.config(state="disabled")
        if CUSTOM_TKINTER_AVAILABLE:
            self.progress_bar.set(0)
        else:
            self.progress_var.set(0)
        self.status_var.set("Обработка видео...") if not CUSTOM_TKINTER_AVAILABLE else self.status_label.configure(text="Обработка видео...")

        processing_thread = threading.Thread(target=self.process_video_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def process_video_thread(self):
        """Обработка видео в отдельном потоке"""
        try:
            self.log_message("Начинаем обработку видео...")
            self.log_message(f"Вход: {self.input_file}")
            self.log_message(f"Выход: {self.output_file}")

            # Запускаем обработку
            process_video(self.input_file, self.output_file, self.progress_callback, self.log_callback)

            # Обработка завершена
            self.root.after(0, self.processing_finished)

        except Exception as e:
            error_msg = f"Ошибка обработки: {str(e)}"
            self.log_message(error_msg)
            self.root.after(0, lambda: self.processing_error(error_msg))

    def progress_callback(self, progress):
        """Callback для обновления прогресса"""
        if CUSTOM_TKINTER_AVAILABLE:
            self.progress_bar.set(progress / 100)
        else:
            self.progress_var.set(progress)

    def log_callback(self, message):
        """Callback для добавления сообщений в лог"""
        self.log_message(message)

    def processing_finished(self):
        """Обработка завершена успешно"""
        self.processing = False
        self.process_btn.configure(state="normal") if CUSTOM_TKINTER_AVAILABLE else self.process_btn.config(state="normal")
        if CUSTOM_TKINTER_AVAILABLE:
            self.progress_bar.set(1.0)
            self.status_label.configure(text="Обработка завершена!")
            self.open_output_btn.configure(state="normal")
        else:
            self.progress_var.set(100)
            self.status_var.set("Обработка завершена!")
            self.open_output_btn.config(state="normal")

        self.log_message("✅ Обработка завершена успешно!")
        self.log_message(f"Результат сохранен: {self.output_file}")

        messagebox.showinfo("Готово!", f"Видео обработано!\nРезультат: {self.output_file}")

    def processing_error(self, error_msg):
        """Ошибка обработки"""
        self.processing = False
        self.process_btn.configure(state="normal") if CUSTOM_TKINTER_AVAILABLE else self.process_btn.config(state="normal")
        status_text = "Ошибка обработки"
        if CUSTOM_TKINTER_AVAILABLE:
            self.status_label.configure(text=status_text)
        else:
            self.status_var.set(status_text)

        messagebox.showerror("Ошибка", error_msg)

    def open_output(self):
        """Открыть папку с результатом"""
        if self.output_file and os.path.exists(self.output_file):
            output_dir = os.path.dirname(self.output_file)
            if sys.platform == "win32":
                os.startfile(output_dir)
            else:
                subprocess.run(["xdg-open", output_dir])
        else:
            messagebox.showwarning("Предупреждение", "Файл результата не найден!")

def main():
    """
    Основная функция программы
    """
    # Проверяем аргументы командной строки
    if len(sys.argv) == 2:
        # Консольная версия
        print("ESP для видео")
        print("=" * 30)

        # Получаем путь к видео из аргументов
        video_path = sys.argv[1].strip()

        # Проверяем существование файла
        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return

        # Определяем путь для вывода
        video_dir = os.path.dirname(video_path) or "."
        output_path = os.path.join(video_dir, "out.mp4")

        print(f"Обрабатываем: {video_path}")
        print(f"Вывод: {output_path}")

        # Обрабатываем видео
        try:
            process_video(video_path, output_path)
        except Exception as e:
            print(f"Ошибка обработки: {e}")
    else:
        # GUI версия
        if CUSTOM_TKINTER_AVAILABLE:
            root = ctk.CTk()
        else:
            root = tk.Tk()

        app = ESPDetectorGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()
