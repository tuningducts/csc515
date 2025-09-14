import numpy as np
import cv2 
from pathlib import Path
import os

def clear_screen():
    os.system('cls'if os.name == 'nt' else 'clear')

clear_screen()
#Set constants 
BASE_DIR = Path(__file__).resolve().parent
ASSETS_PATH = 'assets'
FILENAME = 'shutterstock130285649--250.jpg'
IMAGE_PATH = f"{BASE_DIR}/{ASSETS_PATH}/{FILENAME}"
DEFAULT_COPY_FILENAME = 'copy.jpg'
DEFAULT_COPY_PATH = f'{BASE_DIR}/{ASSETS_PATH}/{DEFAULT_COPY_FILENAME}'
RETURN_TO_MENU = "Press any key to return to main menu..."
IMAGE = cv2.imread(f'{IMAGE_PATH}')
#validate image was loaded 
if IMAGE is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")

def show_image():
    cv2.imshow("Numbers", IMAGE)
    print("press any key to close window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # workaround for issue with window not closing properly on MacOS 

def set_path() -> Path:
    target_path = input("Enter the path to save the image (default - assets/copy.jpg): ") or f"{DEFAULT_COPY_PATH}"
    # sanitize input
    target_path = Path(target_path.strip().strip('"').strip("'"))
    return target_path
    

def copy_image(target_path: Path):   
    #create path if it doesn't exist
    if target_path.is_dir() or target_path.suffix == "":
       target_path = target_path / DEFAULT_COPY_FILENAME
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        target_path = target_path.with_suffix(".jpg")


    copied = cv2.imwrite(str(target_path), IMAGE)
    if not copied:
        raise IOError(f"Failed to write image to {target_path}")
    else:
        print(f"image saved to {target_path}")    

def menu():
    while True:
        clear_screen()
        print("======MAIN MENU======")
        print("1. Display Image")
        print("2. Copy Image")
        print("3. Exit")

        choice = input("Select an option (1-3): ")
       
        # try:
        # #     choice = int(choice)
        # except ValueError:
        #     print("Invalid input. Please enter a number between 1 and 3.")
            # continue 
        if choice == '1':
            clear_screen()
            show_image()
        elif choice == '2':
            clear_screen()
            target = set_path()
            copy_image(target)
            input(RETURN_TO_MENU)
        elif choice == '3':
            clear_screen()
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 3.")



if __name__ == "__main__":
    menu()

        
    
