import struct


# From http://monkeythinkmonkeycode.com/mnist_decoding/


def read_images(images_name):
    # Return an array of flattened images
    f = open(images_name, "rb")
    ds_images = []
    # Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
    mw_32bit = f.read(4)  # magic word
    n_numbers_32bit = f.read(4)  # number of images
    n_rows_32bit = f.read(4)  # number of rows of each image
    n_columns_32bit = f.read(4)  # number of columns of each image

    # convert it to integers ; '&gt;i' for big endian encoding
    mw = struct.unpack('>I', mw_32bit)[0]
    n_numbers = struct.unpack('>I', n_numbers_32bit)[0]
    n_rows = struct.unpack('>I', n_rows_32bit)[0]
    n_columns = struct.unpack('>I', n_columns_32bit)[0]

    try:
        for i in range(n_numbers):
            image = []
            for r in range(n_rows):
                for l in range(n_columns):
                    byte = f.read(1)
                    pixel = struct.unpack('B', byte)[0]
                    image.append(pixel / 255)
            ds_images.append(image)

    finally:
        f.close()
    return ds_images


def read_labels(labels_name):
    # return an array of labels
    f = open(labels_name, "rb")
    ds_labels = []
    # Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
    mw_32bit = f.read(4)  # magic word
    n_numbers_32bit = f.read(4)  # nunber of labels

    # convert it to integers ; '&gt;i' for big endian encoding
    mw = struct.unpack('>i', mw_32bit)[0]
    n_numbers = struct.unpack('>i', n_numbers_32bit)[0]

    try:
        for i in range(n_numbers):
            byte = f.read(1)
            label = struct.unpack('B', byte)[0]
            ds_labels.append(label)

    finally:
        f.close()
    return ds_labels


def read_dataset(images_name, labels_name):
    # reads an image-file and a labels file, and returns an array of tuples of
    # (flattened_image, label)
    images = read_images(images_name)
    labels = read_labels(labels_name)
    assert len(images) == len(labels)
    return zip(images, labels)
