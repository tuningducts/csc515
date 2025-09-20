import cv2

ASSETS_FOLDER = 'assets'
IMAGE_PATH = f'{ASSETS_FOLDER}/cat.jpg'


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception(f"Failed to load {path}")
    return img

def write_check(ok, path):
    if not ok:
        raise Exception(f"Failed to write image to {path}") 
    else:
        print(f"Image written to {path}")
    
def split_channels(img):
    rgb = { 'blue': img[:,:,0], 'green': img[:,:,1], 'red':img[:,:,2] }
    return rgb
    
def alternative_split_channels(img):
    blue, green, red = cv2.split(img)
    return { 'Green': green, 'Red': red, 'Blue': blue}  
 
def save_channels(rgb):
    for key, value in rgb.items():
        cv2.imwrite(f"{ASSETS_FOLDER}/{key}.jpg", value)
        print(f"Saved {key} channel image with shape: {value.shape}")

def show_channel_images(rgb):
    y = 100
    x = 100
    offset = 50 
    for i, (key, value) in enumerate(rgb.items()):
        channel_image = load_img(f"{ASSETS_FOLDER}/{key}.jpg")
        cv2.imshow(f"{key} image", channel_image)
        cv2.moveWindow(f"{key} image", x + i*offset, y + i*offset)

def merge_channels(rgb, file):
    images_to_merge =[]
    for key, value in rgb.items():
        images_to_merge.append(cv2.imread(f"{ASSETS_FOLDER}/{key}.jpg", cv2.IMREAD_GRAYSCALE))
    merged_image = cv2.merge(images_to_merge)
    ok = cv2.imwrite(f"{ASSETS_FOLDER}/{file}", merged_image)
    write_check(ok, f"{ASSETS_FOLDER}/{file}")
    print(f"Merged image channels and saved as {file}")



if __name__ == '__main__':
    print("This program loads an image, splits it into its RGB channels, saves each channel as a seperate image,\
    and then merges the channels back together and displays the original, the seperate channel images, and the merged image.")
    img = load_img(IMAGE_PATH)
    cv2.imshow("Oringinal Image", img)
    cv2.moveWindow("Oringinal Image", 50, 50)
    print("showing original image.")
    rgb = split_channels(img)
    grb = alternative_split_channels(img)
    save_channels(rgb)
    show_channel_images(rgb)
    print("showing channel images.")
    merge_channels(rgb, "merged_image.jpg")
    merge_channels(grb, "reversed_image.jpg")
    merged_image = load_img(f"{ASSETS_FOLDER}/merged_image.jpg")
    reversed_image = load_img(f"{ASSETS_FOLDER}/reversed_image.jpg")
    cv2.imshow("Merged Image", merged_image)
    cv2.moveWindow("Merged Image", 250, 250)
    cv2.imshow("Reversed Image", reversed_image)
    cv2.moveWindow("Reversed Image", 300, 300)
    print("showing merged image. Press any key to exit. ")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

