import cv2
from flask import Flask, render_template, request, redirect, url_for, session
from roboflow import Roboflow
from PIL import Image
import os
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import shutil

app = Flask(__name__)

rf = Roboflow(api_key="ZQL02CEemrLEUVRBugfz")
project = rf.workspace("mangroves-detection").project("bakawan-patrol-v.3-flgtt")
model = project.version(2).model

new_size = (256, 256)
final_resize = (1920, 1080)



class ImagePredictor:
    def __init__(self, model, new_size=(256, 256), final_resize=(1920, 1080)):
        self.model = model
        self.new_size = new_size
        self.final_resize = final_resize

    def predict(self, input_file_path, output_file_path):
        # open the image to be predicted
        image = Image.open(input_file_path)

        # resizing the image
        resized_image = image.resize(self.new_size)
        # saving the resized image to be accessed later
        resized_image.save('Bakawan-1/Resize/Bakawan-1-2.jpg')

        # the resized image goes through the model
        predicted = self.model.predict('Bakawan-1/Resize/Bakawan-1-2.jpg')
        # the predicted image get saved
        predicted.save(output_file_path)

        # opening the predicted image
        predicted_image = Image.open(output_file_path)
        # resizing the predicted image
        final_resized_image = predicted_image.resize(self.final_resize)
        # saving the predicted image
        final_resized_image.save('Bakawan-1/Detection_segmentation/Prediction_Bakawan-1.jpg')


class ImageComparator:
    def __init__(self, model, new_size=(256, 256), final_resize=(1920, 1080)):
        self.model = model
        self.new_size = new_size
        self.final_resize = final_resize

    def predict(self, input_file_path, output_file_path):
        # open the image to be predicted
        image = Image.open(input_file_path)

        # resizing the image
        resized_image = image.resize(self.new_size)
        # saving the resized image to be accessed later
        resized_image.save('Bakawan-2/Resize/Bakawan-2-2.jpg')

        # the resized image goes through the model
        predicted = self.model.predict('Bakawan-2/Resize/Bakawan-2-2.jpg')
        # the predicted image get saved
        predicted.save(output_file_path)

        # opening the predicted image
        predicted_image = Image.open(output_file_path)
        # resizing the predicted image
        final_resized_image = predicted_image.resize(self.final_resize)
        # saving the predicted image
        final_resized_image.save('Bakawan-2/Detection_segmentation/Prediction_Bakawan-2.jpg')


# convert the image to black and white
class ImageConverter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(self.image_path)
        self.gray_img = self.img.convert('L')
        self.gray_array = np.array(self.gray_img)
        self.height, self.width = self.gray_array.shape

    def convert_to_bw(self):
        bw_array = np.zeros((self.height, self.width), dtype=np.uint8)
        bw_array[self.gray_array > 127] = 255
        bw_img = Image.fromarray(bw_array)
        bw_img.save('Bakawan-1/B&W/Bakawan-1b&w.jpg')


# convert the image to black and white
class ImageConverter_2:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = Image.open(self.image_path)
        self.gray_img = self.img.convert('L')
        self.gray_array = np.array(self.gray_img)
        self.height, self.width = self.gray_array.shape

    def convert_to_bw(self):
        bw_array = np.zeros((self.height, self.width), dtype=np.uint8)
        bw_array[self.gray_array > 127] = 255
        bw_img = Image.fromarray(bw_array)
        bw_img.save('Bakawan-2/B&W/Bakawan-2b&w.jpg')


# count the white pixels in an image
class ImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path

        # Open the image file and resize
        self.new_size = (256, 256)
        self.img = Image.open(self.image_path)
        self.image = self.img.resize(self.new_size)

    def analyze_white_area(self):
        # Convert the image to a NumPy array
        img_arr = np.array(self.image)

        # Threshold the image to create a binary image
        thresh = 200
        binary_img = np.where(img_arr > thresh, 1, 0)

        # Count the number of white pixels in the binary image
        num_white_pixels = np.sum(binary_img)

        # Calculate the area of the white region in pixels
        white_area = num_white_pixels * (self.image.width * self.image.height) / img_arr.size
        print("White area: {:.2f}%".format(white_area))

        return white_area


class ImageComparison:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.similarity = None  # Initialize similarity as None

    def compare_images(self):
        # Load images using OpenCV
        image1 = cv2.imread(self.image1_path)
        image2 = cv2.imread(self.image2_path)

        # Resize images to 1920x1080
        target_size = (1920, 1080)
        image1 = cv2.resize(image1, target_size)
        image2 = cv2.resize(image2, target_size)

        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Create SIFT object
        sift = cv2.SIFT_create()

        # Detect and compute keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

        # Create FLANN Matcher object
        matcher = cv2.FlannBasedMatcher()

        # Perform KNN matching
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Filter good matches using the Lowe's ratio test
        good_matches = []
        for match1, match2 in matches:
            if match1.distance < 0.75 * match2.distance:
                good_matches.append(match1)

        # Estimate homography
        src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Draw matching features on the images
        matches_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

        # Save the image with matching features
        cv2.imwrite('matching_features.jpg', matches_image)

        # Warp image1 to image2 using the estimated homography
        warped_image1 = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

        # Compute similarity score
        self.similarity = self.compute_similarity(image2, warped_image1)

        ## Display the images and similarity score
        # plt.figure(figsize=(12, 6))
        # plt.subplot(131), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Mangroves Before')
        # plt.subplot(132), plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)), plt.title('Mangroves After')
        # lt.subplot(133), plt.imshow(cv2.cvtColor(warped_image1, cv2.COLOR_BGR2RGB)), plt.title(
        # f'Mangroves After (Warped)\nSimilarity: {self.similarity:.2f}%')
        # plt.show()

        return warped_image1

    def compute_similarity(self, image1, image2):
        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Compute the Structural Similarity Index (SSIM)
        ssim = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCOEFF_NORMED)

        # Normalize the similarity score between 0 and 1
        similarity = (ssim[0][0] + 1) / 2

        return similarity


def crop_square_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image based on the bounding rectangle
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure secret key

# Dictionary to store username and password combinations
users = {}


# Login route
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and users[username] == password:
            session["username"] = username
            return redirect("/home")
        else:
            return render_template("login.html", error="Invalid USERNAME or PASSWORD!")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if len(password) < 8:
            error = 'Password should be at least 8 characters long.'
            return render_template('signup.html', error=error)

        if username in users:
            error = 'You already have an account.'
            return render_template('signup.html', error=error)
        else:
            # Add the new user to the dictionary (simulating user registration)
            users[username] = password
            return render_template('success.html', username=username)

    return render_template('signup.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Save the file to the server
        input_file = os.path.join('static/uploads', file.filename)
        file.save(input_file)
        predictor = ImagePredictor(model)
        predictor.predict(input_file, 'Bakawan-1/Prediction/Predicted-Bakawan-1.jpg')
        # convert to b&w
        converter = ImageConverter('Bakawan-1/Detection_segmentation/Prediction_Bakawan-1.jpg')
        converter.convert_to_bw()
        # analyze white area
        analyzer = ImageAnalyzer('Bakawan-1/B&W/Bakawan-1b&w.jpg')
        white_area = analyzer.analyze_white_area()
        white = (white_area / (256 * 256)) * 100
        formatted_percentage = "{:.0f}".format(white)

        # Code snippet to move files
        # Specify the source file path
        source_file = 'Bakawan-1/Detection_segmentation/Prediction_Bakawan-1.jpg'

        # Specify the destination file path
        destination_file = 'static/uploads/Prediction_Bakawan-1.jpg'

        # Transfer the file to the destination directory
        shutil.copyfile(source_file, destination_file)

        # Optionally, remove the source file
        os.remove(source_file)

        # Redirect to the predicted image page
        return redirect(url_for('predicted_image', filename='Prediction_Bakawan-1.jpg'))
    else:
        return render_template('index.html')


@app.route('/predicted/<filename>', methods=['GET', 'POST'])
def predicted_image(filename):
    if request.method == 'POST':
        return redirect(url_for('compare_image'))
    return render_template('predicted.html', image_file=filename)


@app.route('/compare', methods=['GET', 'POST'])
def compare_image():
    global white_area
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']

        f1_path = 'static/images/' + secure_filename(f1.filename)
        f2_path = 'static/images/' + secure_filename(f2.filename)

        f1.save(f1_path)
        f2.save(f2_path)

        # loading the images
        image1_path = f1_path
        image2_path = f2_path

        image4_path = f1_path
        image3_path = f2_path

        # making use of the IComparison class
        comparator = ImageComparison(image1_path, image2_path)
        warped_image1 = comparator.compare_images()

        comptor = ImageComparison(image3_path, image4_path)
        warped_image2 = comptor.compare_images()

        # compute similarity
        sim1 = comparator.similarity
        sim_1_rounded_up = round(sim1, 2)
        print("Similarity 1: ", sim_1_rounded_up)

        sim2 = comptor.similarity
        sim_2_rounded_up = round(sim2, 2)
        print("Similarity 1: ", sim_2_rounded_up)

        sim = abs(sim1 / sim2)
        mis = abs(sim2 / sim1)
        print("Sim: {:.2f}".format(sim))
        print("Mis: {:.2f}".format(mis))

        result_alert = ''  # Initialize result_alert variable
        suggestion = ''  # Initialize suggestion variable

        # sim1 calculation
        if 0.90 <= sim_1_rounded_up <= 1.10 and 0.90 <= sim <= 1.10:
            cv2.imwrite('Bakawan/warped_image_1.jpg', warped_image1)
            print('warped_image1 saved')
            image_path = 'Bakawan/warped_image_1.jpg'
            cropped_image1 = crop_square_image(image_path)
            cv2.imwrite("Bakawan/Cropped Image 1.jpg", cropped_image1)
            print('Cropped_image_1 saved')

            comp1 = ImageComparison(image1_path, "Bakawan/Cropped Image 1.jpg")
            comp1.compare_images()
        else:
            print('Similarity index_1 is too low')

        # sim2 calculation
        if 0.90 <= sim_2_rounded_up <= 1.10 and 0.90 <= mis <= 1.10:
            cv2.imwrite('Bakawan/warped_image_2.jpg', warped_image2)
            print('warped_image2 saved')
            image_path = 'Bakawan/warped_image_2.jpg'
            cropped_image2 = crop_square_image(image_path)
            cv2.imwrite("Bakawan/Cropped Image 2.jpg", cropped_image2)
            print('Cropped_image_2 saved')

            comp2 = ImageComparison(image3_path, "Bakawan/Cropped Image 2.jpg")
            comp2.compare_images()
        else:
            print('Similarity index_2 is too low')

        if not (0.90 <= sim_1_rounded_up <= 1.10 and 0.90 <= sim <= 1.10 and 0.90 <= sim_2_rounded_up <= 1.10 and 0.90 <= mis <= 1.10):
            print('Proceed is True')

            predictor = ImagePredictor(model)
            predictor.predict(f1_path, 'Bakawan-1/Prediction/Predicted-Bakawan-1.jpg')  ### change f1 path or not
            # convert to b&w
            converter = ImageConverter('Bakawan-1/Detection_segmentation/Prediction_Bakawan-1.jpg')
            converter.convert_to_bw()
            # analyze white area
            analyzer = ImageAnalyzer('Bakawan-1/B&W/Bakawan-1b&w.jpg')
            white_area = analyzer.analyze_white_area()

            comparator = ImageComparator(model)
            comparator.predict(f2_path, 'Bakawan-2/Prediction/Predicted-Bakawan-2.jpg')  ### change f2 path

            # convert to b&w
            converter2 = ImageConverter_2('Bakawan-2/Detection_segmentation/Prediction_Bakawan-2.jpg')
            converter2.convert_to_bw()

            analyzer_2 = ImageAnalyzer('Bakawan-2/B&W/Bakawan-2b&w.jpg')
            white_area_2 = analyzer_2.analyze_white_area()

            if white_area > white_area_2:
                result_alert = 'Alert! The mangrove forest has experienced a significant decrease in size or density between the two time periods.'
                suggestion = 'Investigate the potential causes of the decline, such as deforestation, climate change, or human activities, and take necessary actions to address them.'
            elif white_area < white_area_2:
                result_alert = 'Mangrove growth is detected! The mangrove forest has experienced a significant increase in size or density between the two time periods.'
                suggestion = 'Consider implementing measures to protect the newly established mangrove areas from potential threats, such as pollution, coastal development, or unsustainable fishing practices.'

        elif 0.90 <= sim_1_rounded_up <= 1.10 and 0.90 <= sim <= 1.10 and 0.90 <= sim_2_rounded_up <= 1.10 and 0.90 <= mis <= 1.10:
            result_alert = 'Nothing has changed! There is no significant difference in the mangrove forest between the two images.'
            suggestion = 'Continue monitoring the mangrove forest to detect any potential changes in the future.'

            print('Proceed is False')

        # Access the values of result_alert and suggestion outside the conditional blocks
        print(result_alert)
        print(suggestion)

        return render_template('result.html', suggestion1=suggestion, result1=result_alert, file1=f1_path, file2=f2_path)
    return render_template('compare_image.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
