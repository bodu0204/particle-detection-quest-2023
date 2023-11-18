import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

def normalize_map(map):
    from PIL import Image

    # リサイズ後のサイズを指定
    resize_shape = (28, 28)
    # マップの次元を取得
    len_y, len_x = map.shape

    # マップの中心y座標とx座標を取得
    y_add = len_y // 2 + len_y % 2
    x_add = len_x // 2 + len_x % 2

    # 0の値のインデックスを取得
    y_indices, x_indices = np.where(map == 0)
    # 0の値を置換
    map[y_indices, x_indices] = map[(y_indices + y_add) % len_y, (x_indices + x_add) % len_x]
    # リサイズし、1を減算
    resized_map = Image.fromarray(map - 1.0).resize(resize_shape)
    return np.asarray(resized_map, dtype=np.float16)

def preprocess_map(train_df, normalize_map):
    # データの正規化
    normalized_train_maps = np.array([normalize_map(x) for x in train_df['waferMap']])

    # データ拡張（90度回転、水平反転など）
    normalized_train_maps = np.concatenate((normalized_train_maps, np.rot90(normalized_train_maps, k=2, axes=(1, 2))), axis=0)
    normalized_train_maps = np.concatenate((normalized_train_maps, np.rot90(normalized_train_maps, k=1, axes=(1, 2))), axis=0)
    normalized_train_maps = np.concatenate((normalized_train_maps, np.swapaxes(normalized_train_maps, 1, 2)), axis=0)

    # データの形状を変更
    normalized_train_maps = normalized_train_maps.reshape(normalized_train_maps.shape + (1,))

    return normalized_train_maps
    
def create_model(input_shape, num_classes):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, activation=tf.nn.relu, kernel_size=(3,3), padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(3,3), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, dtype='float32'),
    ])
    return model

def calculate_class_weights(train_labels):
    from sklearn.utils.class_weight import compute_class_weight
    # クラスの重みを計算
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(train_labels), 
                                         y=train_labels)
    # クラスの重みを辞書型に変換
    return dict(enumerate(class_weights))

def plot_confusion_matrix_and_accuracy(y_true, y_pred, classes):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Confusion matrixの計算
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # ヒートマップとしてプロット
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 各クラスごとの正確さと最も間違えやすいクラスを表示
    print("\nClass Accuracy and Most Common Errors:")
    for i, class_name in enumerate(classes):
        accuracy = cm[i, i] / cm[i, :].sum()
        print(f"{class_name}: Accuracy: {accuracy * 100:.2f}%")
        
        # 最も間違えやすいクラスを特定
        error_indices = cm[i, :].argsort()[-2:-1] if accuracy < 1 else []
        for error_index in error_indices:
            error_rate = cm[i, error_index] / cm[i, :].sum()
            error_class = classes[error_index]
            print(f"    Most common error: Mistaken for {error_class} ({error_rate * 100:.2f}%)")

def solution(x_test_df, train_df):
    import tensorflow as tf
    failure_types = list(train_df['failureType'].unique())

    # 前処理
    normalized_train_maps = preprocess_map(train_df, normalize_map)
    # データ拡張を行う場合はtrain_labelsを変更する必要がある
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)

    class_weights = calculate_class_weights(train_labels)

    model = create_model(normalized_train_maps[0].shape, len(failure_types))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(normalized_train_maps, train_labels, epochs=2, class_weight=class_weights)

    normalized_test_maps = np.array([normalize_map(x) for x in x_test_df['waferMap']])
    normalized_test_maps = normalized_test_maps.reshape(normalized_test_maps.shape + (1,))

    predictions = tf.nn.softmax(model.predict(normalized_test_maps, batch_size=32)).numpy().astype(np.float32)
    answer = [failure_types[x.argmax()] for x in predictions]

    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)

# 以下は編集しないでください
# データのインポート
df=pd.read_pickle("../work/input/LSWMD_25519.pkl")

# テスト用と学習用のデータを作成（テストする際は、random_stateの値などを編集してみてください）
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

y_test_df = test_df[['failureType']]
x_test_df = test_df.drop(columns=['failureType'])

# solution関数を実行
user_result_df = solution(x_test_df, train_df)
plot_confusion_matrix_and_accuracy(y_test_df['failureType'], user_result_df['failureType'], df['failureType'].unique())

average_accuracy = 0
# ユーザーの提出物のフォーマット確認
if type(y_test_df) == type(user_result_df) and y_test_df.shape == user_result_df.shape:
    # 平均精度の計算
    accuracies = {}
    for failure_type in df['failureType'].unique():
        y_test_df_by_failure_type = y_test_df[y_test_df['failureType'] == failure_type]
        user_result_df_by_failure_type = user_result_df[y_test_df['failureType'] == failure_type]
        matching_rows = (y_test_df_by_failure_type == user_result_df_by_failure_type).all(axis=1).sum()
        accuracies[failure_type] = (matching_rows/(len(y_test_df_by_failure_type)))
    
    average_accuracy = sum(accuracies.values())/len(accuracies)
print(f"平均精度：{average_accuracy*100:.2f}%")