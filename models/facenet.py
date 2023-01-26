import tensorflow as tf
import os
import requests
import zipfile as zf

class Facenet(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Load the model from the given path.
        The model is a pre-trained facenet model.

        Returns:
            The loaded model.
        """
        # Download the model if it doesn't exist
        if not os.path.exists(self.model_path) or len(os.listdir(self.model_path)) == 0:
            self.download_model()
        # Load the model
        model_name = os.listdir(self.model_path)[0]
        path = os.path.join(self.model_path, model_name)
        print(os.path.join(path, model_name+".pb"))
        model = tf.saved_model.load(path)
        return model

    def download_model(self):
        """
        Download the facenet model from the given url.
        """
        # Download the model
        url = "https://github.com/davegabe/Suspect-Identification-CCTV/releases/download/pretrained-models/facenet-pretrained-VGGFace2.zip"
        file_name = "facenet-pretrained-VGGFace2.zip"
        os.makedirs(self.model_path, exist_ok=True)
        print("Downloading facenet model...")
        r = requests.get(url, allow_redirects=True)
        open(self.model_path + file_name, 'wb').write(r.content)
        print("Download complete.")
        
        # Unzip the downloaded file
        print("Unzipping facenet model...")
        with zf.ZipFile(self.model_path + file_name, 'r') as zip_ref:
            zip_ref.extractall(self.model_path)
        print("Unzipping complete.")
        
        # Delete the zip file
        os.remove(self.model_path + file_name)


if __name__ == "__main__":
    facenet = Facenet("../facenet/")
    print(facenet.model)