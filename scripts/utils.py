import cv2


def rotateImage(image, angle):
    """Rotate image by given angle clockwise and return processed image."""

    height = image.shape[0]
    width = image.shape[1]
    height_big = height * 2
    width_big = width * 2
    image_big = cv2.resize(image, (width_big, height_big))
    image_center = (width_big/2, height_big/2)

    # rotation center
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
    result = cv2.warpAffine(image_big, rot_mat, (width_big, height_big), flags=cv2.INTER_LINEAR)

    new_height = result.shape[0]
    new_width = result.shape[1]
    new_image_center = (new_height / 2, new_width / 2)
    if height > width:
        result = result[int(new_image_center[0]-(height/2)):int(new_image_center[0]+(height/2)),
                int(new_image_center[1]-(width/2)):int(new_image_center[1]+(width/2))]
    else:
        result = result[int(new_image_center[0]-(width/2)):int(new_image_center[0]+(width/2)),
                int(new_image_center[1]-(height/2)):int(new_image_center[1]+(height/2))]
    return result


def rectangleContains(bbox, x, y):
    """Check if rectangle contains given point."""

    if x is None or y is None:
        return 0
    else:
        if bbox[0] < x and x < bbox[2] and bbox[1] < y and y < bbox[3]:
            return 1
        else:
            return 0