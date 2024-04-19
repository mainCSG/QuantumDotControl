import scipy
import scipy.optimize
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import numpy as np

from src.inference import *  

def calc_dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2

    d = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return d

def calc_slope(p1, p2):
    x1,y1 = p1
    x2,y2 = p2

    m = (y2-y1)/(x2-x1)
    return m

def calculate_perpendicular_line(point1, point2, point):
    """Calculate the perpendicular line from a point to a line defined by two points."""
    # Calculate the slope of the original line
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0] + 1e-10)  # Avoid division by zero

    # Calculate the perpendicular slope
    slope_perpendicular = -1 / slope

    # Calculate the x-coordinate of the intersection point
    x_intersection = (point[1] - point1[1] + slope * point1[0] - slope_perpendicular * point[0]) / (slope - slope_perpendicular)

    # Calculate the y-coordinate of the intersection point
    y_intersection = slope * (x_intersection - point1[0]) + point1[1]

    return (x_intersection, y_intersection)

def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def extract_baseline(image, mask, tolerance):

    poly = np.array(mask_to_polygon(mask, tolerance)).reshape(-1,2)
    hull = scipy.spatial.ConvexHull(poly)
    hull_indices = hull.vertices
    convex_hull_points = np.array([poly[i] for i in hull_indices])
    xs, ys = convex_hull_points[:,0],convex_hull_points[:,1] 
    plt.imshow(image)
    plt.fill(xs, ys, 'r-', alpha=0.3)  # Connect the last point to the first to close the polygon
    plt.scatter(xs ,ys)
    plt.title("Extreme Points via Convex Hull")
    plt.show()
    corners_dict = {}
    for i in range(len(convex_hull_points)+1):
        index1, index2 = i % len(convex_hull_points), (i+1) % len(convex_hull_points)
        p1, p2 = convex_hull_points[i % len(convex_hull_points)], convex_hull_points[(i+1) % len(convex_hull_points)]
        d = calc_dist(p1,p2)
        m = calc_slope(p1,p2)
        corners_dict[(index1,index2)] = {'dist': d, 'slope': m}

    corners_dict_sorted = dict(sorted(corners_dict.items(), key=lambda item: item[1]['dist']))

    min_dist = corners_dict_sorted[list(corners_dict_sorted.keys())[0]]['dist']
    max_dist = corners_dict_sorted[list(corners_dict_sorted.keys())[-1]]['dist']

    edges = canny(image,sigma=2, mask=mask)

    lines = probabilistic_hough_line(edges, line_length=int(np.average([min_dist, max_dist])/2))

    slopes = []
    yints = []
    points =[]
    for line in lines:
        p0, p1 = line
        slope = (p1[1]-p0[1])/(p1[0]-p0[0])
        yint = p1[1] - slope * p1[0]
        if slope == 0 or (p1[0]-p0[0]) == 0 :
            continue
        yints.append(yint)
        slopes.append(slope)
        points.append((p0,p1))

        plt.scatter((p0[0], p1[0]), (p0[1], p1[1]), marker='*',s=50)
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

    median_index = np.argsort(np.array(yints))[len(np.array(yints))//2]
    median_point = points[median_index]
    plt.imshow(edges)
    plt.show()

    hist, bins = np.histogram(slopes, bins=20, density=True)

    # Find bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit histogram data to a Gaussian distribution
    (mu, sigma) = scipy.stats.norm.fit(image)

    popt, _ = scipy.optimize.curve_fit(gaussian, bin_centers, hist, p0=[mu, sigma])
    # Print mean and standard deviation of the fitted Gaussian
    print("Fitted Mean:", popt[0])
    print("Fitted Standard Deviation:", popt[1])
    plt.hist(slopes, bins=15, density=True, alpha=0.7, color='blue', label='Histogram')

    # Plot fitted Gaussian
    plt.plot(bin_centers, gaussian(bin_centers, *popt), 'r--', label='Fitted Gaussian')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Detected Slopes')
    # plt.legend()
    plt.show()

    for indices, info in corners_dict_sorted.items():

        slope=info['slope']
        # Calculate the PDF of the fitted Gaussian at the given slope
        pdf_at_slope = scipy.stats.norm.pdf(slope, loc=popt[0], scale=popt[1])
        info['prob'] = pdf_at_slope
        # Print the probability

    # Sort the dictionary based on 'prob' and then 'dist' keys in descending order
    sorted_data = sorted(corners_dict_sorted.items(), key=lambda x: (np.abs(x[1]['prob'])**2 * x[1]['dist']), reverse=True)

    # Get the key with the highest prob and highest dist
    key_with_highest_prob_and_dist = sorted_data[0][0]

    # Print the key
    print("CHull indices with the highest prob and highest dist:", key_with_highest_prob_and_dist)

    best_p1, best_p2 = convex_hull_points[key_with_highest_prob_and_dist[0]], convex_hull_points[key_with_highest_prob_and_dist[1]]
    base_line = np.array([best_p1, best_p2])

    plt.imshow(image)
    plt.plot(base_line[:,0], base_line[:,1])
    plt.show()

    return base_line

def extract_detuning_path(image, point, baseline):

    # Example points
    point1 = baseline[0]
    point2 = baseline[1]

    intersection_point1 = calculate_perpendicular_line(point1, point2, point)

    plt.figure(figsize=(8, 6))
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='white', linestyle='--', linewidth=2, label='Baseline')
    plt.plot([point[0], intersection_point1[0]], [point[1], intersection_point1[1]], color='white', linestyle='--', linewidth=2, label='Detuning Axis')
    plt.scatter(point[0], point[1], color='white',marker='*',s=50)

    plt.imshow(image)
    # plt.legend(loc='best')
    plt.show()
