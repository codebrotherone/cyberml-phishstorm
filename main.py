from tqdm import tqdm

# custom funcs
from utils.funcs import *


def train(epochs=10, batch_size=20):
    from tqdm import tqdm

    random_dim = 100

    # Get the training and testing data
    # for generator and discriminator
    x_train = scaled_xtrain

    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = create_gen_model()
    discriminator = create_disc_model()

    gan = create_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(int(batch_count))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            url_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake url features
            gen_urls = generator.predict(noise)
            X = np.concatenate([url_batch, gen_urls])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator or generator
            # based on bellow trainable attr()
            discriminator.trainable = False
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            gan.train_on_batch(noise, y_gen)


if __name__ == '__main__':
    train(400, 128)