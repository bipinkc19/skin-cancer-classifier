
class Basic_model(tf.keras.Model):
    def __init__(self, device='gpu:0'):
        super(Basic_model, self).__init__()
        self.device = device
        self._input_shape = [-1, 299, 299, 3]

        self.image_module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
        self.conv1 = tf.layers.Conv2D(8, 3,
                                  padding='same',
                                  activation=tf.nn.relu)
        self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = tf.layers.Conv2D(16, 2,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(50, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.5)
        self.fc2 = tf.layers.Dense(7)
    
    def call(self, x):
        x = tf.reshape(x, self._input_shape)
        # x = self.max_pool2d(self.conv1(x))
        # x = self.max_pool2d(self.conv2(x))
        x = image_module(x)
        # x = tf.layers.flatten(x)
        # x = self.dropout(self.fc1(x))
        return self.fc2(x)

def loss_fn(model, x, y):
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=model(x), labels=y))

def get_accuracy(model, x, y_true):
    logits = model(x)
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

model = Basic_model()
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
dataset = DataLoad('./HAM10000/data.tfrecords', EPOCHS, BATCH_SIZE).return_dataset()

for i, batch in enumerate(dataset):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, batch['image'], batch['label'])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
    if i % 100 == 0:
        acc = get_accuracy(model, batch['image'], batch['label']).numpy()
        print("Iteration {}, loss: {:.3f}, train accuracy: {:.2f}%".format(i , loss_fn(model, batch['image'], batch['label']).numpy(), acc*100))
