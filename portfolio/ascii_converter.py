# string contains characters from the "lightest" to "darkest", the dark characters taking up more space have a higher index in the string
ASCII_STRING = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def advancedGreyscale(img, r, g, b):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    x = len(red)
    y = len(red[0,:])
    greyArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            greyArr[i][j] = (int(red[i][j] * r) + int(green[i][j] * g) + int(blue[i][j] * b)) // 3

    return greyArr

def convert_to_ascii(img):
        ascii_matrix = []
        for row in img:
            ascii_row = []
            for pixel in row:
                ascii_row.append(ASCII_STRING[int(pixel / 255 * len(ASCII_STRING)) - 1])
            ascii_matrix.append(ascii_row)

        return ascii_matrix