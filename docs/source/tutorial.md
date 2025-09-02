# Tutorial

Welcome to Metoryx!

Metoryx is a neural network library built on top of JAX.
This tutorial will guide you through the basics of using Metoryx to build and train neural networks, using MNIST classification as an example.

## Installation and Imports

You can install Metoryx using pip:

```bash
pip install metoryx
```

In this tutorial, we import the following libraries:

```python
import itertools

import jax
import jax.random as jr
import optax
import metoryx as mx

# For data loading
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Define MNIST Dataset Loader

This tutorial uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).
We can easily load this dataset using TensorFlow Datasets.
Here's a function to load and preprocess the MNIST dataset:

```python
def get_dataset(num_epochs, batch_size, split):
    """Load the specified split of MNIST dataset."""
    ds = tfds.load("mnist", split=split)
    ds = ds.map(
        lambda item: {
            "image": tf.cast(item["image"], tf.float32) / 255.0,
            "label": item["label"],
        }
    )
    ds = ds.repeat(num_epochs)
    if split == "train":
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size, drop_remainder=(split == "train"))
    ds = ds.prefetch(1)
    return ds
```

## Define Convolutional Neural Network

Now, define a simple convolutional neural network (CNN) for classifying MNIST images.
In Metoryx, models are defined by subclassing `mx.Module` and implementing the `__call__` method.

```python
class ConvNet(mx.Module):
    """A simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = mx.Conv(1, 32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.conv2 = mx.Conv(32, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.fc1 = mx.Dense(7 * 7 * 64, 128)
        self.fc2 = mx.Dense(128, 10)

    def __call__(self, x, is_training):
        x = self.conv1(x)  # Input shape is expected to be (batch, height, width, channels)
        x = mx.relu(x)
        x = mx.max_pool(x, kernel_size=(2, 2), strides=(2, 2))
        x = self.conv2(x)
        x = mx.relu(x)
        x = mx.max_pool(x, kernel_size=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = mx.dropout(x, rate=0.5, is_training=is_training)
        x = self.fc1(x)
        x = mx.relu(x)
        return self.fc2(x)

# Instantiate the model.
net = ConvNet()
```

## Transform the Model

This is a crucial step in Metoryx. In Metoryx, your model instance is not directly callable like in some other libraries.
Instead, you need to transform it into a pair of functions using `mx.transform`.

```python
init_fn, apply_fn = mx.transform(net)
```

`init_fn` initializes the model parameters and states, while `apply_fn` applies the model to inputs using the given parameters and states. These functions are pure and can be used with JAX transformations like `jax.jit`, `jax.vmap`, and `jax.pmap`.

## Define Training and Evaluation Steps

Next, we define the training and evaluation steps using `jax.jit` for efficiency.
If you are familiar with JAX, you will notice that the training step is similar to standard JAX code.

```python
@jax.jit
def train_step(rng, params, state, opt_state, batch):
    def loss_fn(params):
        # Combine params and state into a single variables dictionary
        variables = {"params": params, **state}
        # new_variables contains a PyTree with the same structure as variables, but with updated arrays (e.g., batch norm stats)
        logits, new_state = apply_fn(variables, rng, batch["image"], is_training=True)
        new_state.pop("params")

        # Compute loss and accuracy
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        accuracy = (logits.argmax(axis=-1) == batch["label"]).mean()
        log_dict = {"loss": loss, "accuracy": accuracy}

        return loss, (new_state, log_dict)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (new_state, log_dict)), grads = grad_fn(params)

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, new_opt_state, log_dict


@jax.jit
def eval_step(params, state, batch):
    variables = {"params": params, **state}
    logits, _ = apply_fn(
        variables, None, batch["image"], is_training=False
    )  # If rng is not needed, None can be passed
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    accuracy = (logits.argmax(axis=-1) == batch["label"]).mean()
    return {"loss": loss, "accuracy": accuracy}
```

## Initialize Parameters and Optimizer

Prior to training, we need to initialize the model parameters and the optimizer state.

```python
rng = jr.PRNGKey(42)  # Set random seed
rng, init_rng = jr.split(rng)
state = init_fn(init_rng)  # State contains 'params' and other states (e.g., batch norm stats)
params = state.pop("params")  # Separate params from other states

optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)
opt_state = optimizer.init(params)
```

## Let's Train the Model!

Now, we are ready to train the model!
The training loop iterates over the dataset, calling `train_step` for each batch and updating the model parameters.
To track the training progress, we use `mx.utils.AverageMeter` to compute average metrics over each epoch.

```python
num_epochs = 10
batch_size = 64

num_images = 60000
num_steps_per_epoch = num_images // batch_size

train_ds = get_dataset(num_epochs, batch_size, split="train")
train_iter = iter(train_ds.as_numpy_iterator())
for epoch in range(num_epochs):
    meter = mx.utils.AverageMeter()
    for batch in itertools.islice(train_iter, num_steps_per_epoch):
        rng, rng_apply = jr.split(rng)
        params, state, opt_state, metrics = train_step(rng_apply, params, state, opt_state, batch)
        meter.update(metrics, n=len(batch["image"]))
    print(meter.compute())
```

Output should be similar to:

```
{'accuracy': 0.9218916755602988, 'loss': 0.24640471186576335}
{'accuracy': 0.9744530416221985, 'loss': 0.08127296438466523}
{'accuracy': 0.9802061099252934, 'loss': 0.062307235655616355}
{'accuracy': 0.9840081376734259, 'loss': 0.051378412921629694}
{'accuracy': 0.9864594450373533, 'loss': 0.04249529187615241}
{'accuracy': 0.9875433564567769, 'loss': 0.038763899686518936}
{'accuracy': 0.988977454642476, 'loss': 0.03436332501764504}
{'accuracy': 0.9900780416221985, 'loss': 0.03115085096888914}
{'accuracy': 0.9897778815368197, 'loss': 0.030075993747828107}
{'accuracy': 0.9914120864461046, 'loss': 0.02679101809990476}
```

## Evaluation

After training, evaluate the model on the test dataset.
The evaluation loop is similar to the training loop, but we use `eval_step` and do not update the model parameters.

```python
test_ds = get_dataset(1, batch_size, split="test")
meter = mx.utils.AverageMeter()
for batch in test_ds.as_numpy_iterator():
    metrics = eval_step(params, state, batch)
    meter.update(metrics, n=len(batch["image"]))
print("Test:", meter.compute())
```

Output should be similar to:

```
Test: {'accuracy': 0.9925, 'loss': 0.020609416964557023}
```

## Conclusion

In this tutorial, we covered the basics of using Metoryx to build and train a convolutional neural network for MNIST classification. We defined a model using `mx.Module`, transformed it into pure functions, and implemented training and evaluation steps using JAX. We also utilized `mx.utils.AverageMeter` to track and summarize metrics during training and evaluation.