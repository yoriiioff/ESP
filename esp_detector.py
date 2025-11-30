import cv2
# Note: Using opencv-python-headless for compatibility
import os
import sys
import tempfile
import subprocess
from ultralytics import YOLO
import torch

# Проверяем доступность moviepy
try:
    import moviepy
    from moviepy.config import change_settings
    # Настраиваем moviepy для использования локального ffmpeg
    ffmpeg_path = r"N:\ai maked ai\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        change_settings({"FFMPEG_BINARY": ffmpeg_path})
        print("Настроен локальный ffmpeg для moviepy")
    VideoFileClip = moviepy.VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("MoviePy не доступен, видео будет сохранено без аудио")

def detect_and_draw_esp(frame, model, class_names):
    """
    Обнаруживает объекты и рисует ESP-подобные элементы
    """
    results = model(frame)

    # Классы для обнаружения (только существующие в YOLOv8 COCO dataset)
    target_classes = {
        'person': 'Person',
        'traffic light': 'Traffic Light',
        'tv': 'Monitor',
        'laptop': 'Monitor',
        'car': 'Car',
        'truck': 'Car',
        'bus': 'Car',
        'motorcycle': 'Car',
        'bicycle': 'Car',
        'potted plant': 'Tree',  # Растения в горшках вместо деревьев
        'chair': 'Chair',
        'couch': 'Couch',
        'dining table': 'Table',
        'bed': 'Bed',
        'toilet': 'Toilet'
    }

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

                # Проверяем, является ли класс целевым
                # Используем более низкий порог уверенности для некоторых объектов
                confidence_threshold = 0.2 if class_name == 'traffic light' else 0.4
                if class_name in target_classes and conf > confidence_threshold:
                    # Рисуем красный прямоугольник
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Добавляем подпись
                    label = target_classes[class_name]
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame, detected_classes

def process_video(video_path, output_path):
    """
    Обрабатывает видео и сохраняет результат с сохранением аудио
    """
    # Загружаем модель YOLOv8
    print("Загрузка модели YOLOv8...")
    model = YOLO('yolov8n.pt')  # Используем nano версию для скорости

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {video_path}")
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

    print(f"Обработка видео: {total_frames} кадров")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр
        processed_frame, detected_classes = detect_and_draw_esp(frame, model, None)

        # Записываем обработанный кадр
        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Показываем прогресс каждые 30 кадров
            progress = (frame_count / total_frames) * 100
            print(".1f")
            if detected_classes:
                print(f"Обнаруженные объекты: {', '.join(detected_classes[:5])}")  # Показываем первые 5

    # Освобождаем ресурсы OpenCV
    cap.release()
    out.release()

    # Теперь объединяем видео с аудио с помощью ffmpeg
    try:
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
                print("Аудио успешно объединено с видео!")
            else:
                print(f"Ошибка ffmpeg: {result.stderr}")
                # Fallback: копируем видео без звука
                import shutil
                shutil.copy2(temp_video_path, output_path)
        else:
            print("FFmpeg не найден, видео сохранено без аудио")
            import shutil
            shutil.copy2(temp_video_path, output_path)

    except Exception as e:
        print(f"Ошибка при объединении аудио: {e}")
        # Fallback: копируем видео без звука
        import shutil
        shutil.copy2(temp_video_path, output_path)
        print("Видео сохранено без аудио")

    # Удаляем временный файл
    try:
        os.unlink(temp_video_path)
    except:
        pass

    print(f"Обработка завершена! Результат сохранен в: {output_path}")

def main():
    """
    Основная функция программы
    """
    print("ESP Детектор для видео")
    print("=" * 30)

    # Проверяем аргументы командной строки
    if len(sys.argv) != 2:
        print("Использование: python esp_detector.py <путь_к_видео>")
        print("Пример: python esp_detector.py video.mp4")
        return

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

if __name__ == "__main__":
    main()
