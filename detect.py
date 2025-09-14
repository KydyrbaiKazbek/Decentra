import argparse
import os
from ultralytics import YOLO
from pathlib import Path

def detect_damage(image_path, model_path='weights/best.pt'):
    """
    Запускает модель YOLOv8 для детекции повреждений на изображении.

    :param image_path: Путь к изображению для анализа.
    :param model_path: Путь к файлу с весами модели (.pt).
    """
    # Проверяем, существует ли файл с моделью
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден по пути: {model_path}")
        return

    # Проверяем, существует ли изображение
    if not os.path.exists(image_path):
        print(f"Ошибка: Изображение не найдено по пути: {image_path}")
        return

    print("Загрузка модели...")
    # Загружаем нашу обученную модель
    model = YOLO(model_path)

    print(f"Анализ изображения: {image_path}")
    # Запускаем предсказание
    # save=True - сохраняет изображение с результатами
    # project='results' - папка для сохранения
    # name='detection' - подпапка для сохранения
    results = model.predict(source=image_path, save=True, project='results', name='detection', exist_ok=True)

    # Получаем путь к сохраненному файлу
    output_path = Path(f"results/detection/{Path(image_path).name}")
    print(f"✅ Готово! Результат сохранен в файл: {output_path.resolve()}")

if __name__ == '__main__':
    # Создаем парсер для аргументов командной строки
    parser = argparse.ArgumentParser(description="Детекция повреждений автомобиля на изображении с помощью YOLOv8.")
    
    # Добавляем обязательный аргумент - путь к изображению
    parser.add_argument(
        'image_path', 
        type=str, 
        help="Путь к изображению для анализа."
    )

    args = parser.parse_args()
    
    # Вызываем функцию детекции
    detect_damage(args.image_path)
