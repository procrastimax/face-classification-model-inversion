import tensorflow as tf

import os

num_skipped = 0
folder_path = "face-images"
for foldername in os.listdir("face-images"):
    fname = os.path.join(folder_path, foldername)
    if os.path.isdir(fname):
        for image in os.listdir(fname):
            image_path = os.path.join(fname, image)
            try:
                print(f"opening {image_path}")
                fobj = open(image_path, "rb")
                is_jfif = tf.compat.as_bytes("PNG") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                print(f"Found {image_path} as corrupted image")
                # Delete corrupted image
                os.remove(image_path)

print("Deleted %d images" % num_skipped)
