import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to fit a circle for the droplet's boundary
def circle_function(x, a, b, r):
    return b + np.sqrt(r**2 - (x - a)**2)

# Function to calculate contact angle from the tangent slope
def calculate_contact_angle(x, y, a, b, r):
    slope_left = - (x[0] - a) / np.sqrt(r**2 - (x[0] - a)**2)
    slope_right = - (x[-1] - a) / np.sqrt(r**2 - (x[-1] - a)**2)

    angle_left = np.degrees(np.arctan(slope_left))
    angle_right = np.degrees(np.arctan(slope_right))

    contact_angle = 180 - abs(angle_left - angle_right)
    return contact_angle

# Main code to process the image
def process_droplet_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found at the specified path.")

    # Threshold the image to isolate the droplet
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the droplet
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")

    # Assume the largest contour is the droplet
    droplet_contour = max(contours, key=cv2.contourArea)

    # Fit a bounding ellipse to the droplet
    (x, y), (width, height), angle = cv2.fitEllipse(droplet_contour)

    # Fit a circle to the lower boundary (for contact angle)
    droplet_points = droplet_contour.squeeze()
    x_data = droplet_points[:, 0]
    y_data = droplet_points[:, 1]

    # Sort points along x-axis
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]

    # Fit a circle to the lower part of the droplet (contact region)
    params, _ = curve_fit(circle_function, x_data, y_data, p0=[x, y, width / 2])
    a, b, r = params

    # Calculate the contact angle
    contact_angle = calculate_contact_angle(x_data, y_data, a, b, r)

    # Show results
    print(f"Droplet Width: {width:.2f} pixels")
    print(f"Droplet Height: {height:.2f} pixels")
    print(f"Contact Angle: {contact_angle:.2f} degrees")

    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_img, cmap='gray')
    plt.plot(x_data, y_data, 'ro', label='Droplet Boundary')

    # Plot the fitted circle
    x_fit = np.linspace(x_data[0], x_data[-1], 500)
    y_fit = circle_function(x_fit, *params)
    plt.plot(x_fit, y_fit, 'b-', label='Fitted Circle')

    plt.title("Droplet Analysis")
    plt.legend()
    plt.show()

# Example Usage
image_path = "droplet.png"  # Replace with your droplet image path
process_droplet_image(image_path)
