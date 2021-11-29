from flask import Flask, request, Response,jsonify
# from sample import caption
import torch
# import matplotlib.pyplot as plt
import numpy as np 
# import argparse
# import pickle 
# import os
from torchvision import transforms 
# from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import pyttsx3
from flask import make_response, send_file
import nltk
import pickle
import argparse
from collections import Counter
app = Flask(__name__)
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class Vocabulary(object):
#     """Simple vocabulary wrapper."""
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.idx = 0

#     def add_word(self, word):
#         if not word in self.word2idx:
#             self.word2idx[word] = self.idx
#             self.idx2word[self.idx] = word
#             self.idx += 1

#     def __call__(self, word):
#         if not word in self.word2idx:
#             return self.word2idx['<unk>']
#         return self.word2idx[word]

#     def __len__(self):
#         return len(self.word2idx)
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def caption(vocab, imagepath):
    
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    # with open('data/vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # Load vocabulary wrapper
    

    # Build models
    encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('models/encoder-5-3000.pkl'))
    decoder.load_state_dict(torch.load('models/decoder-5-3000.pkl'))

    # Prepare an image
    image = load_image(imagepath, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    # image = Image.open('png/example.png')
    # plt.imshow(np.asarray(image))
    return (sentence)
# global vocab
# with open('data/vocab.pkl', 'rb') as f:
#         vocab = pickle.load(f)
@app.route('/')
def index():
    return "Backend running on port 5000"

# @app.route('/predict', methods=['POST','GET'])
# def test():
#     # with open('data/vocab.pkl', 'rb') as f:
#     #     vocab = pickle.load(f)
    
#     if request.method == 'POST':
#         file = request.files['image']
#         img = Image.open(file.stream)
#         img = img.save("img1.jpeg")
#         text=caption('img1.jpeg')
#         text=text[8:-5]

#         return {"predicted": text}
#     else :
#         return "I'm alive!"
if __name__ == "__main__":
    
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    @app.route('/predict', methods=['POST','GET'])
    def test():
        # with open('data/vocab.pkl', 'rb') as f:
        #     vocab = pickle.load(f)
        
        if request.method == 'POST':
            file = request.files['image']
            img = Image.open(file.stream)
            img = img.save("img1.jpeg")
            text=caption(vocab,'img1.jpeg')
            text=text[8:-5]

            return {"predicted": text}
        else :
            return "I'm alive!"
    
    app.run(debug=True,port=8080)
