def load_data(path: str, data: pd.DataFrame) -> tuple[str, int]:
    images_path: list[str] = []
    steering: list[int] = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        images_path.append(os.path.join(path, indexed_data[0]))
        steering.append(float(indexed_data[1]))
    images_path = np.asarray(images_path)
    steering = np.asarray(steering)
    return images_path, steering


# Augment
def augment_image(img_path: str, steering: int):
    img = mpimg.imread(img_path)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={
                         "x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def pre_process(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img