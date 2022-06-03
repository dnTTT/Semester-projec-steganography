import hashlib
import os
from Cryptodome.Cipher import AES
import easygui
import binascii
import numpy as np
from PIL import Image
import sys, random

from easygui import multenterbox

EOF_MARKER = '$eof!'.encode('utf-8')
hexa_EOF = EOF_MARKER.hex()
confs = {'RGB': [0, 3], 'RGBA': [1, 4]}

def calculate_image_min_size(number_of_bits):
    min_pixels = number_of_bits // 3  # 1 pixel = 3 bits stored
    return min_pixels


def get_num_rand(used_pixels, num_of_pixels):
    number = random.randint(0, (num_of_pixels-1))
    while number in used_pixels:
        number = random.randint(0, (num_of_pixels-1))
    used_pixels.add(number)
    return number

def select_option():
    print("---------------------------------------")
    print("[*] Options:")
    print("  [1] - Encrypt")
    print("  [2] - Decrypt")
    print("---------------------------------------")
    selection = input("Selection: ")
    return selection

def encrypt_aes(data,password):
    salt = os.urandom(32)

    key = hashlib.scrypt(password, salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    encrypted_message = salt + cipher.nonce + tag + ciphertext
    encrypted_message_hexa = binascii.hexlify(encrypted_message)
    return encrypted_message_hexa

def decrypt_aes(hexa_image, password):

    # IMPROVE - For lack of better way of splitting the string
    # It's written to a file and then the file is deleted
    temp_file = "encryptedfile.bin"
    file_out = open(temp_file, "wb")
    encrypted_message = binascii.unhexlify(hexa_image)
    file_out.write(encrypted_message)
    file_out.close()
    file_in = open(temp_file, "rb")
    salt, nonce, tag, ciphertext = [file_in.read(x) for x in (32, 16, 16, -1)]
    key = hashlib.scrypt(password, salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    cipher = AES.new(key, AES.MODE_GCM, nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    file_in.close()
    os.remove(temp_file)
    return data

def open_image():
    try:
        image = easygui.fileopenbox(filetypes=[".png"])
        open_image = Image.open(image, 'r')
        return open_image
    except:
        return None

def write_to_image(seed, input_data, output_file):
    bytes_written = 0
    used_pixels = set()

    # Pseudo-random number generator with a seed
    random.seed(seed)

    img = open_image()
    if img is None:
        print("Image could not be opened")
        sys.exit()
    width, height = img.size

    # Returns the contents of this image as a sequence object containing pixel values.
    matrix = np.array(list(img.getdata()))

    # Get the configuration, if it's RGB or RGBA
    conf = confs[img.mode]
    num_of_pixels = matrix.size // conf[1]
    # Add a marker to when searching for the message on the image, it knows where it ends
    input_data += hexa_EOF.encode()
    input_data = input_data.decode('utf-8')
    # Converts input data and end marker to binary.
    binary_enc = "".join([format(ord(ch), "08b") for ch in str(input_data)])
    #print(f"string: {input_data.decode('utf-8')}")
    # Get the min size of the image needed to write the input data
    min_size = calculate_image_min_size(len(binary_enc))
    if min_size >= num_of_pixels:
        print("ERROR: The image is not big enough to properly write the message")
        sys.exit(1)

    start_pixel = get_num_rand(used_pixels, num_of_pixels)
    while bytes_written != len(input_data):
        # for each bit on each byte of the message
        bit_index = 0
        while bit_index != 8:
            current_pixel = matrix[start_pixel]

            for rgb_color in range(conf[0], conf[1]):
                if bit_index == 8:
                    break
                # Color RGB example: [144 , 74, 67]
                current_color = current_pixel[rgb_color]
                """
                    Bitwise calculation
                    Bitwise AND operator: Returns 1 if both the bits are 1 otherwise returns 0.
                """
                # Get the least significant bit
                lsb = current_color & 1

                # If that least significant bit is different from the bit to change
                if lsb != int(binary_enc[(bytes_written * 8) + bit_index]):
                    current_color = current_color >> 1
                    current_color = current_color << 1
                    if lsb == 0:
                        current_color = current_color | 1

                    current_pixel[rgb_color] = current_color
                bit_index += 1
            start_pixel = get_num_rand(used_pixels, num_of_pixels)
        bytes_written += 1
        bit_index = 0
        start_pixel = get_num_rand(used_pixels, num_of_pixels)

    out_img = Image.fromarray(np.uint8(matrix.reshape(height, width, conf[1])), img.mode)
    out_img.save(output_file)
    print("Message hidden in the image with success")

def read_from_image(seed):
    img = open_image()
    if img == None:
        print("Image could not be opened")
        sys.exit()
    width, height = img.size

    matrix = np.array(list(img.getdata()))
    conf = confs[img.mode]
    num_of_pixels = matrix.size // conf[1]
    random.seed(seed)
    used_pixels = set()
    start_pixel = get_num_rand(used_pixels, num_of_pixels)
    bit_index = 7
    byte = 0
    message = ""
    end_of_message = False
    while (end_of_message == False):
        while (bit_index >= 0):
            current_pixel = matrix[start_pixel]

            for c in range(conf[0], conf[1]):
                if bit_index >= 0:

                    byte += (current_pixel[c] & 1) << bit_index
                    bit_index -= 1
                else:
                    break
            start_pixel = get_num_rand(used_pixels, num_of_pixels)
        if start_pixel >= num_of_pixels:
            break
        # Decoded one byte
        message += chr(byte)

        if message[-len(hexa_EOF):] == hexa_EOF:
            end_of_message = True
        byte = 0
        bit_index = 7
        start_pixel = get_num_rand(used_pixels, num_of_pixels)

    if end_of_message == False:
        print("Nothing found in this image")
        return ""
    else:
        return message[:len(message) - len(hexa_EOF)]

def message_gui():
    return easygui.enterbox("Message to encrypt")

def password_gui():
    return easygui.passwordbox("Enter password")

def seed_gui():
    return easygui.enterbox("Enter a seed (important to decipher text)")

def output_gui():
    output = easygui.enterbox("Name of the output file (no extension needed)")
    return output + ".png"


def All_in_one_GUI():
    msg = "Enter your personal information"
    title = "Credit Card Application"
    fieldNames = ["Message", "Password", "Seed", "Output file"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = multenterbox(msg, title, fieldNames)

    # make sure that none of the fields was left blank
    while 1:
        if fieldValues is None:
            break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break  # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)

    print("Reply was: %s" % str(fieldValues))
    data_to_encrypt = fieldValues[0]
    password = fieldValues[1]
    seed = fieldValues[2]
    output_file = fieldValues[3]


if __name__ == '__main__':
    #All_in_one_GUI()
    selected_func = select_option()

    if selected_func == "1":
        data_to_encrypt = message_gui()
        password = password_gui()
        seed = seed_gui()
        output_file = output_gui()
        input_message = encrypt_aes(data_to_encrypt.encode(), password.encode())
        write_to_image(seed, input_message, output_file)
    elif selected_func == "2":
        password = password_gui()
        seed = seed_gui()
        encrypted_message = read_from_image(seed)
        decrypted_message = decrypt_aes(encrypted_message, password.encode())
        print(decrypted_message)
    else:
        print("[*] Invalid option")



