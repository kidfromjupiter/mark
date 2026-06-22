import cv2
import numpy as np

# We designed the figma mcq answer sheet to be exactly 842, 595. 
# each aruco marker is 40px on figma. 
WIDTH = 802
HEIGHT = 555




def normalise_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    ID_MAP = {
        0: "TL",  # Top-Left Marker ID
        1: "TR",  # Top-Right Marker ID
        3: "BR",  # Bottom-Right Marker ID
        2: "BL"   # Bottom-Left Marker ID
    }

    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) >= 4:
        # Flatten IDs for easier mapping
        detected_ids = ids.flatten()
        
        # Create a dictionary to hold our found coordinates
        found_points = {}
        
        for i, marker_id in enumerate(detected_ids):
            if marker_id in ID_MAP:
                corner_role = ID_MAP[marker_id]
                # Use the center point of the marker as the corner point
                marker_center = np.mean(corners[i][0], axis=0)
                found_points[corner_role] = marker_center
    
        # 2. Verify all 4 corners were successfully detected
        if len(found_points) == 4:
            # Construct the source array in a strictly guaranteed order
            src_points = [
                found_points["TL"],
                found_points["TR"],
                found_points["BR"],
                found_points["BL"]
            ]
            src_mat = np.array(src_points, dtype=np.float32)
    
            # 3. Define the static destination array
            dst_mat = np.array([
                [0, 0],       # TL
                [WIDTH - 1, 0],   # TR
                [WIDTH - 1, HEIGHT - 1], # BR
                [0, HEIGHT - 1]    # BL
            ], dtype=np.float32)
    
            # 4. Transform safely!
            M = cv2.getPerspectiveTransform(src_mat, dst_mat)
            warped = cv2.warpPerspective(img, M, (WIDTH, HEIGHT))

            return warped

    return None


if __name__ == "__main__":
    IMG_PATH = "pic2.jpeg"
    img = cv2.imread(IMG_PATH)

    if img is not None:
        k = normalise_img(img)
        cv2.imshow("result",k)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




