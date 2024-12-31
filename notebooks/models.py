def inception_module(input_tensor, filters_1x1, filters_3x3, filters_5x5, filters_7x7):
    conv1x1 = layers.Conv1D(filters_1x1, kernel_size=1, padding='same', activation='relu')(input_tensor)

    conv3x3 = layers.Conv1D(filters_3x3, kernel_size=3, padding='same', activation='relu')(input_tensor)

    conv5x5 = layers.Conv1D(filters_5x5, kernel_size=5, padding='same', activation='relu')(input_tensor)

    conv7x7 = layers.Conv1D(filters_7x7, kernel_size=7, padding='same', activation='relu')(input_tensor)

    max_pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)

    output = layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, conv7x7, max_pool])

    return output

def create_model(input_length, input_kernel=50, inception1=10, inception2=20, inception3=30):
    input_layer = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(filters=10,
                      kernel_size=input_kernel,
                      strides=1, padding='same',
                      kernel_initializer='he_normal')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = inception_module(x,
                         filters_1x1=inception1,
                         filters_3x3=inception1,
                         filters_5x5=inception1,
                         filters_7x7=inception1
                         )

    x = inception_module(x,
                         filters_1x1=inception2,
                         filters_3x3=inception2,
                         filters_5x5=inception2,
                         filters_7x7=inception2
                         )

    x = inception_module(x,
                         filters_1x1=inception3,
                         filters_3x3=inception3,
                         filters_5x5=inception3,
                         filters_7x7=inception3
                         )

    x = layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(360, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(180, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output_layer = layers.Dense(1)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model