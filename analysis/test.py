import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

def normalize_map(map, resize_shape=(64,64)):
    import numpy as np
    from PIL import Image

    (len_y,len_x) = map.shape
    y_add = len_y // 2 + len_y % 2
    x_add = len_x // 2 + len_x % 2
    for y in range(len_y):
        for x in range(len_x):
            if map[y][x] == 0:
                map[y][x] = map[(y + y_add) % len_y][(x + x_add) % len_x]
    return np.asarray(Image.fromarray(map - 1.0).resize(resize_shape))
    

def create_model(input_shape=(227, 227, 3), num_classes=1000):
    import tensorflow as tf

    layers = tf.keras.layers
    return tf.keras.models.Sequential([
        layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(5,5), padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(5,5), padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(num_classes, activation=tf.nn.softmax),
    ])

def show_img(image):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def solution(x_test_df, train_df):
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight

    resize_shape = (64,64)

    failure_types = list(train_df['failureType'].unique())
    # 前処理
    normalized_train_maps = np.array([normalize_map(x) for x in train_df['waferMap']])
    print(normalized_train_maps.shape)
    normalized_train_maps = np.concatenate((normalized_train_maps, np.rot90(normalized_train_maps, k=2, axes=(1, 2))), axis=0)
    normalized_train_maps = np.concatenate((normalized_train_maps, np.rot90(normalized_train_maps, k=1, axes=(1, 2))), axis=0)
    normalized_train_maps = np.concatenate((normalized_train_maps, np.swapaxes(normalized_train_maps, 1, 2)), axis=0)
    normalized_train_maps = normalized_train_maps.reshape(normalized_train_maps.shape + (1,))
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)
    show_img(normalized_train_maps[0])
    print(normalized_train_maps.shape)
    print(train_labels.shape)
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(train_labels), 
                                         y=train_labels)
    class_weights = dict(enumerate(class_weights))
    model = create_model(input_shape=normalized_train_maps[0].shape,num_classes=len(failure_types))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    
    model.fit(normalized_train_maps, train_labels, epochs=2, class_weight=class_weights)
    
    normalized_test_maps = np.array([normalize_map(x) for x in x_test_df['waferMap']])
    normalized_test_maps = normalized_test_maps.reshape(normalized_test_maps.shape + (1,))
    test_index = [x for x in x_test_df['waferIndex']]
    predictions = model.predict(normalized_test_maps)
    answer = [failure_types[x.argmax()] for x in predictions]
    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)

# データのインポート
df=pd.read_pickle("../work/input/LSWMD_25519.pkl")

# テスト用と学習用のデータを作成（テストする際は、random_stateの値などを編集してみてください）
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

y_test_df = test_df[['failureType']]
x_test_df = test_df.drop(columns=['failureType'])

# solution関数を実行
user_result_df = solution(x_test_df, train_df)

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