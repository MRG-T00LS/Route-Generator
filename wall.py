from flask import Flask, request, send_file, render_template
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_route', methods=['POST'])
def generate_route():
    # Load the image from the request
    img_array = np.fromstring(request.files['image'].read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of colors for hold detection (e.g., yellow, orange, red)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to detect holds
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask_yellow, mask_orange)
    mask = cv2.bitwise_or(mask, mask_red)

    # Apply morphological operations to refine the hold detection
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours of the holds
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (e.g., noise, shadows)
    hold_locations = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if area > 100 and aspect_ratio > 0.5 and aspect_ratio < 2:
            hold_locations.append((x, y, w, h))

    # Sort holds by y-coordinate (top to bottom)
    hold_locations.sort(key=lambda x: x[1])

    # Generate a route by selecting a subset of holds
    route_holds = [hold_locations[0]]  # Start at the top hold
    for i in range(4):  # Select 4 more holds for the route
        min_dist = float('inf')
        next_hold = None
        for hold in hold_locations:
            if hold not in route_holds:
                dist = np.linalg.norm(np.array(hold[:2]) - np.array(route_holds[-1][:2]))
                if dist < min_dist:
                    min_dist = dist
                    next_hold = hold
        route_holds.append(next_hold)

    # Circle the holds on the original image
    for hold in route_holds:
        x, y, w, h = hold
        cv2.circle(img, (x + w // 2, y + h // 2), 10, (0, 255, 0), 2)

    # Save the output image to a file
    cv2.imwrite('route_image.jpg', img)

    # Return the output image as a response
    return send_file('route_image.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)