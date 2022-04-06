import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.simpledialog as sdg
import numpy as np
import cv2
import os
from naive_algorithm import naive_and_homogenize
from dlt_algorithm import dlt_and_homogenize
from dlt_modified_algorithm import dlt_modified_and_homogenize
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

left_coords = []
right_coords = []


def on_image_click(event):
    global left_coords
    global app
    if app.selected_image is None:
        app.show_info("No image selected")
    elif event.inaxes is not None:
        ix = round(event.xdata, 0)
        iy = round(event.ydata, 0)
        if len(left_coords) < app.points_number:
            left_coords.append((ix, iy))
            app.update_cordinates()
            app.scatter_coordinates(True, (ix, iy))
        elif len(right_coords) < app.points_number:
            right_coords.append((ix, iy))
            app.update_cordinates()
            app.scatter_coordinates(False, (ix, iy))
        else:
            app.show_info("You exceed the number of picked coordinates. "
                          "If you made a mistake, there's a reset button!")
    else:
        app.show_info("You have clicked outside axes bounds. Try picking the coordinates of the photo?")


class Application:
    def __init__(self, output_path="./"):
        self.top = tk.Tk()
        self.top.resizable(False, False)
        self.top.title("Projective distortion removal tool")

        self.help_active = False

        self.selected_image = None
        self.processed_image = None

        self.points_number = 4

        self.lbl_left_coords_text = tk.StringVar()
        self.lbl_left_coords_text.set("Selected coordinates for the left photo: []")

        self.lbl_right_coords_text = tk.StringVar()
        self.lbl_right_coords_text.set("Selected coordinates for the right photo: []")

        self.frame_1 = tk.Frame(master=self.top)

        self.btn_help = tk.Button(master=self.frame_1, text="Show help",
                                  command=self.on_btn_help_click)
        self.btn_help.grid(padx=20, row=0, column=0)

        self.btn_upload = tk.Button(master=self.frame_1, text="Upload image",
                                    command=self.on_btn_upload_click)
        self.btn_upload.grid(padx=20, row=0, column=1)

        self.btn_reset_img = tk.Button(master=self.frame_1, text="Reset images",
                                       command=self.on_btn_reset_img_click)
        self.btn_reset_img.grid(padx=20, row=0, column=2)

        self.btn_reset_coords = tk.Button(master=self.frame_1, text="Reset coordinates",
                                          command=self.on_btn_reset_coords_click)
        self.btn_reset_coords.grid(padx=20, row=0, column=3)

        self.btn_naive = tk.Button(master=self.frame_1, text="Naive algorithm", command=self.on_btn_naive_click)
        self.btn_naive.grid(padx=(20, 0), row=0, column=4)

        self.btn_dlt = tk.Button(master=self.frame_1, text="DLT", command=self.on_btn_dlt_click)
        self.btn_dlt.grid(row=0, column=5)

        self.btn_n_dlt = tk.Button(master=self.frame_1, text="Normalized DLT", command=self.on_btn_n_dlt_click)
        self.btn_n_dlt.grid(row=0, column=6)

        self.lbl_coords = tk.Label(master=self.frame_1, wraplength=950, textvariable=self.lbl_left_coords_text)
        self.lbl_coords.grid(pady=20, row=1, columnspan=10)
        self.lbl_coords = tk.Label(master=self.frame_1, wraplength=950, textvariable=self.lbl_right_coords_text)
        self.lbl_coords.grid(pady=10, row=2, columnspan=10)

        self.frame_1.pack(padx=10, pady=15)

        self.frame_2 = tk.Frame(master=self.top)
        self.frame_3 = tk.Frame(master=self.top)

        self.btn_quit = tk.Button(master=self.frame_3, text="Quit", command=self.quit)
        self.btn_quit.pack(side='right')

        self.f = plt.Figure(figsize=(14, 6), dpi=100)
        self.left_image = self.f.add_subplot(1, 2, 1)
        self.right_image = self.f.add_subplot(1, 2, 2)

        self.init_plots(True, True)

        plt.show()

        self.f.patch.set_facecolor('#202326')

        self.canvas = FigureCanvasTkAgg(self.f, master=self.frame_2)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.f.canvas.mpl_connect('button_press_event', on_image_click)

        self.frame_2.pack(padx=10, pady=10)
        self.frame_3.pack(padx=10, pady=10)

    def show_info(self, text):
        tk.messagebox.showinfo("Info", text)

    def prompt_integer(self):
        number = sdg.askinteger("Points", "Enter the number of points that you will pick.\n"
                                          "Must be 4 or more. For practical reasons, max is 40.",
                                          parent=self.top,
                                          initialvalue=4,
                                          minvalue=4, maxvalue=40)
        self.points_number = number

    def update_cordinates(self):
        global left_coords
        global right_coords
        self.lbl_left_coords_text.set("Selected coordinates for the left photo: " + str(left_coords))
        self.lbl_right_coords_text.set("Selected coordinates for the right photo: " + str(right_coords))

    def init_plots(self, left, right):
        if left is True:
            self.left_image.cla()
            title_left = self.left_image.set_title("Selected image")
            plt.setp(title_left, color='white')
            self.left_image.axis('off')

        if right is True:
            self.right_image.cla()
            title_right = self.right_image.set_title("Processed image")
            plt.setp(title_right, color='white')
            self.right_image.invert_yaxis()
            self.right_image.axis('off')

    def transform_photo(self, matrix):
        if self.processed_image is not None:
            self.init_plots(False, True)
        img = cv2.imread(self.selected_image)
        height, width, channels = img.shape
        dst = cv2.warpPerspective(img, matrix, (width, height))
        self.processed_image = os.path.split(self.selected_image)[0] + '/dst.bmp'
        cv2.imwrite(self.processed_image, dst)
        self.plot_processed_image()

    def on_btn_naive_click(self):
        global left_coords
        global right_coords
        if self.selected_image is None:
            self.show_info("You'll have to pick the image first.")
            return
        elif self.points_number is None:
            self.show_info("Invalid number of coordinates picked.")
            return
        elif self.points_number > 4:
            self.show_info("Naive algorithm can work only with 4 points!")
            return
        elif len(right_coords) != 4:
            self.show_info("Not enough coordinates picked.")
            return
        matrix = naive_and_homogenize(left_coords, right_coords)
        print("Naive:\n", matrix)
        print()
        if matrix.size == 0:
            self.show_info("The system is not solvable.")
            return
        self.transform_photo(matrix)

    def on_btn_dlt_click(self):
        global left_coords
        global right_coords
        if self.selected_image is None:
            self.show_info("You'll have to pick the image first.")
            return
        if len(left_coords) != self.points_number or len(right_coords) != self.points_number:
            self.show_info("Invalid number of coordinates picked.")
            return
        matrix = dlt_and_homogenize(left_coords, right_coords)
        print("Dlt:\n", matrix)
        print()
        if matrix.size == 0:
            self.show_info("The system is not solvable.")
            return
        self.transform_photo(matrix)

    def on_btn_n_dlt_click(self):
        global left_coords
        global right_coords
        if self.selected_image is None:
            self.show_info("You'll have to pick the image first.")
            return
        if len(left_coords) != self.points_number or len(right_coords) != self.points_number:
            self.show_info("Invalid number of coordinates picked.")
            return
        matrix = dlt_modified_and_homogenize(left_coords, right_coords)
        print("Dlt modified:\n", matrix)
        print()
        if matrix.size == 0:
            self.show_info("The system is not solvable.")
            return
        self.transform_photo(matrix)

    def on_btn_reset_coords_click(self):
        global left_coords
        global right_coords
        if self.selected_image is not None:
            self.prompt_integer()
        left_coords = []
        right_coords = []
        self.update_cordinates()
        self.plot_uploaded_image()

    def on_btn_reset_img_click(self):
        self.selected_image = None
        self.processed_image = None
        self.init_plots(True, True)
        self.on_btn_reset_coords_click()
        self.f.canvas.draw()

    def on_btn_help_click(self):
        if not self.help_active:
            self.help_window = tk.Toplevel(self.top)
            self.help_window.resizable(False, False)
            self.lbl_help = tk.Label(master=self.help_window,
                                     padx=15,
                                     pady=15,
                                     text="This tool makes it possible to remove "
                                          "projective distortion from an image.\n\n"
                                          "How it works:\n"
                                          "1) Upload a .bmp image from your file system. "
                                          "The image will show up on the left side of the "
                                          "window.\n"
                                          "2) Choose the number of coordinates you want algorithm "
                                          "to work with. Default is 4.\n"
                                          "3) First, pick the coordinates that determine "
                                          "distorted surface in the image you've uploaded.\n"
                                          "    Don't worry if you get the coordinates wrong, "
                                          "just use the button to reset the data.\n"
                                          "4) Pick the coordinates on the left photo where you want\n"
                                          "    to project that surface. "
                                          "You might want it to be regularly shaped.\n"
                                          "3) Process image using desired algorithm (3 buttons on the right)\n"
                                          "\nEnjoy!",
                                     justify=tk.LEFT)
            self.help_window.title("Help")
            self.lbl_help.pack()
            self.help_active = True
            self.help_window.protocol("WM_DELETE_WINDOW", self.on_help_closing)

    def on_help_closing(self):
        self.help_window.destroy()
        self.help_active = False

    def on_btn_upload_click(self):
        filename = tk.filedialog.askopenfilename(filetypes=[('BMP FILES', '*.bmp')], title="Choose an image")
        if filename:
            print(type(filename), filename)
            self.selected_image = filename
            self.plot_uploaded_image()
            self.prompt_integer()

    def scatter_coordinates(self, left, A):
        x, y = A
        if left is True:
            self.left_image.scatter(x=[x], y=[y], marker='x', color='blue', alpha=0.7)
        else:
            self.right_image.scatter(x=[x], y=[y], marker='x', color='green', alpha=0.7)

        self.f.canvas.draw()

    def plot_processed_image(self):
        img = mpimg.imread(self.processed_image)
        self.init_plots(False, True)
        self.right_image.imshow(img)
        self.f.canvas.draw()

    def plot_uploaded_image(self):
        self.init_plots(True, True)
        if self.selected_image is not None:
            img = mpimg.imread(self.selected_image)
            img_size = np.shape(img)
            self.right_image.set_xlim([0, img_size[1]])
            self.right_image.set_ylim([0, img_size[0]])
            self.right_image.invert_yaxis()
            self.left_image.imshow(img)
        self.f.canvas.draw()

    def quit(self):
        self.top.destroy()


app = Application()
app.top.mainloop()