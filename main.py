import customtkinter
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
print(tf.__version__)

# Create Model and load the saved model that we train
saved_model_dir = "datatrain/saved_model/my_model"
new_model = tf.saved_model.load(saved_model_dir)

batch_size = 32
img_height = 180
img_width = 180

# Specify the class names
class_names = ['Bacterial_Spot', 'Early_blight', 'Leaf_Mold', 'Powdery_mildew', 'Septoria_leaf_spot']

# Function for browse and display the image
def searchImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="/dataset",title="Select Image",
    filetypes= (("JPG File","*.jpg*"),("PNG file","*.png"), ("All Files", "*.*")))

    my_image = customtkinter.CTkImage(light_image=Image.open(filename),
                                  size=(250, 250))
    image_label.configure(image=my_image)
    return filename

# Function to run the app to recognize the tomato leaf
def runApp():
    global filename
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    predictions = new_model(input_tensor)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This is the image of {} with a {:.2f} %"" confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    print(Output)
    imagePredict.configure(text=Output)

# End of All Function

app = customtkinter.CTk()
customtkinter.set_appearance_mode("system")  # default
app.geometry("1000x800")
app.title("Tomato Leaf Disease Recognition")

labelTitle = customtkinter.CTkLabel(app, fg_color="transparent",text="Tomato Leaf Disease Classification", anchor="center", font=('Arial', 22, "bold"))
labelTitle.place(rely=0.05, relx=0.32)

buttonSelect = customtkinter.CTkButton(app, text="Select File", font=('Arial', 22), width=300, height=40, corner_radius=20, command=searchImage)
buttonSelect.place(rely=0.15, relx=0.35)

image_label = customtkinter.CTkLabel(app, text="")
image_label.place(rely=0.25, relx=0.38)

buttonPredict = customtkinter.CTkButton(app, text="Predict", font=('Arial', 22), width=300, height=40, corner_radius=20, command=runApp)
buttonPredict.place(rely=0.60, relx=0.35)

imagePredict = customtkinter.CTkLabel(app, text="", font=('Arial', 18))
imagePredict.place(rely=0.70, relx=0.25)


app.mainloop()