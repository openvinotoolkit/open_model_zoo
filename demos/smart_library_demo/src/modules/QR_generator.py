import qrcode


class QRgenerator(object):

    # Constructor
    def __init__(self):
        self.generator = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M
        )

    # Creating Qr-code

    def makeQR(self, data):
        
        """Add data to this QR Code."""
        self.generator.data_list.clear()
        self.generator.add_data(data)
        
        """
        Compile the data into a QR Code array.
        :param fit: If ``True`` (or if a size has not been provided), find the
            best fit for the data to avoid data overflow errors.
        """
        self.generator.make(fit=True)

        """ Make an image from the QR Code data."""
        image = self.generator.make_image(fill_color="black", back_color="white")

        return image
