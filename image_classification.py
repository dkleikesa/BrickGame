import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNet
import json
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np

image_dir = '/Users/jianzhangli/src/BrickGame/classBitmap_data'
classed_list = [ "beijing", "baicai", "baiping", "baiqiu", "baocai", "cao", "caoche", "chazi", "daocha", "fangzi", "futou", "hualan", "huapen", "huaping", "huluobo", "huotui", "huoyan", "jiandao", "jitui", "jiucai", "jiutong", "lajiao", "lingdang", "manjing", "maozi", "mianfen", "niupai", "qiezi", "qingjiao", "shoutao", "shuazi", "shuiguan", "shuihu", "shuipen", "shuzhuang", "tieqiao", "wandou", "xiantuan", "xihongshi", "xilanhua", "xinfeng", "xuezi", "yumi", "zhalan"]


def preprocess_images(img_height=224, img_width=224, batch_size=32):
    datagen = ImageDataGenerator(
        validation_split=0.2,  # 使用20%的数据作为验证集
        preprocessing_function = lambda x: tf.image.resize(x, [img_height, img_width])
    )

    train_generator = datagen.flow_from_directory(
        directory=image_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes = classed_list
    )

    validation_generator = datagen.flow_from_directory(
        directory=image_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes = classed_list
    )
    sorted_labels = list(train_generator.class_indices.keys())
    with open('classBitmapModel/class_labels.json', 'w') as json_file:
        json.dump(sorted_labels, json_file)
    print(sorted_labels)
    return train_generator, validation_generator


def build_model_v1(num_classes):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # 冻结基模型

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_v3(num_classes):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # 冻结基模型
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # 假设你的图标图片存放在'path_to_images'文件夹中
    train_data, val_data = preprocess_images()
    use_v3 = True

    if use_v3:
        model = build_model_v3(44)
    else:
        model = build_model_v1(44)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=15
    )
    loss, accuracy = model.evaluate(val_data)
    print(f"Validation accuracy: {accuracy*100:.2f}%")
    if use_v3:
        model.save('classBitmapModel/banzhuan_mobilenetv3_model.h5')
    else:
        model.save('classBitmapModel/banzhuan_mobilenetv1_model.h5')


def convert_model_to_mobile():
    # 加载Keras模型
    model = tf.keras.models.load_model('classBitmapModel/banzhuan_mobilenetv3_model.h5')
    # 设置转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # 选择优化策略，这里使用默认的优化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 进行转换
    tflite_model = converter.convert()
    # 保存TFLite模型
    with open('classBitmapModel/banzhuan_mobilenetv3_model.tflite', 'wb') as f:
        f.write(tflite_model)

def load_and_preprocess_image(image_path):
    # 读取图像文件
    img_raw = tf.io.read_file(image_path)
    # 解码图像文件到张量，自动推断图像类型
    img_tensor = tf.image.decode_image(img_raw, channels=3)
    # 调整图像大小到目标尺寸
    img_resized = tf.image.resize(img_tensor, [224, 224])
    # 扩展维度以适应模型输入：从 [height, width, channels] 到 [batch_size, height, width, channels]
    img_expanded = tf.expand_dims(img_resized, axis=0)
    # 应用预处理函数（根据具体使用的模型进行调整）
    img_preprocessed = preprocess_input(img_expanded)
    return img_preprocessed

def predict_image(image_dir):
    # 文件路径
    filename = 'classBitmapModel/class_labels.json'
    # 打开文件并加载JSON数据
    with open(filename, 'r') as file:
        class_labels = json.load(file)
    print(len(class_labels))
    print(class_labels)

    model = tf.keras.models.load_model('classBitmapModel/banzhuan_mobilenetv3_model.h5')

    rate = 0
    sum = 0
    for root, dirs, files in os.walk(image_dir):
        for image_path in files:
            if not image_path.endswith('.png'):
                continue
            full_image_path = os.path.join(root, image_path)
            print(full_image_path)
            processed_image = load_and_preprocess_image(full_image_path)
            prediction = model.predict(processed_image)
            label = class_labels[np.argmax(prediction)]
            sum = sum + 1
            if label == image_path.split('_')[0]:
                rate = rate + 1
            else:
                print(image_path + "->" + label)

    print('acc:'+str(rate/sum))


if __name__ == '__main__':
    train_model()
    convert_model_to_mobile()
    predict_image(image_dir)
    print('main')

