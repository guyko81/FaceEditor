import pygame
import random
import numpy as np
import cv2
import h5py
from dutil import *

#User constants
device = "cpu"
enc_fname = './model/Encoder.h5'
background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_color = (20, 20, 20)
num_params = 80
input_w = 178
input_h = 218
image_scale = 3
image_padding = 10
slider_w = 15
slider_h = 100
slider_px = 5
slider_py = 10
slider_cols = 20

#Derived constants
slider_w = slider_w + slider_px*2
slider_h = slider_h + slider_py*2
drawing_x = image_padding
drawing_y = image_padding
drawing_w = input_w * image_scale
drawing_h = input_h * image_scale
slider_rows = (num_params - 1) / slider_cols + 1
sliders_x = drawing_x + drawing_w + image_padding
sliders_y = image_padding
sliders_w = slider_w * slider_cols
sliders_h = slider_h * slider_rows
window_w = drawing_w + image_padding*3 + sliders_w
window_h = drawing_h + image_padding*2

#Global variables
prev_mouse_pos = None
mouse_pressed = False
cur_slider_ix = 0
needs_update = True
cur_params = np.zeros((num_params,), dtype=np.float32)
cur_face = np.zeros((3, input_h, input_w), dtype=np.uint8)
rgb_array = np.zeros((input_h, input_w, 3), dtype=np.uint8)

#Keras
print ("Loading Keras...")
import os

print ("Loading Encoder...")
#enc_model = load_model(enc_fname)
print ("Loading Keras...")
import os, math

from keras.initializers import RandomUniform
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model
from keras import backend as K

num_samples = 2000
PARAM_SIZE = 80
LR = 0.01

print ("Building Model...")
model = Sequential()

model.add(Embedding(num_samples, PARAM_SIZE, input_length=1))
model.add(Flatten(name='pre_encoder'))
print (model.output_shape)
assert(model.output_shape == (None, PARAM_SIZE))

model.add(Reshape((1, 1, PARAM_SIZE), name='encoder'))
print (model.output_shape)

model.add(Conv2DTranspose(256, (5, 4)))           #(5, 4)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(256, 4, strides=(1, 1)))     #(8, 7)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(256, 4, strides=(1, 2)))      #(11, 16)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(256, 4, strides=(2, 1)))     #(24, 19)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(128, 4, strides=(2, 2)))     #(50, 40)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(128, 4, strides=(1, 1)))     #(53, 43)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(128, 4, strides=(2, 2)))     #(108, 88)
model.add(Activation("relu"))
print (model.output_shape)

model.add(Conv2DTranspose(3, 4, strides=(2, 2)))      #(218, 178)
model.add(Activation("sigmoid"))
print (model.output_shape)
assert(model.output_shape[1:] == (218, 178, 3))

model.compile(optimizer=Adam(lr=LR), loss='mse')


import h5py
data = h5py.File(enc_fname, 'r')
group = data['model_weights']

model.layers[0].set_weights([group['embedding_1']['embedding_1']['embeddings:0'].value])

weight = [group['conv2d_transpose_1']['conv2d_transpose_1']['kernel:0'].value,
          group['conv2d_transpose_1']['conv2d_transpose_1']['bias:0'].value]
model.layers[3].set_weights(weight)

weight = [group['conv2d_transpose_2']['conv2d_transpose_2']['kernel:0'].value,
          group['conv2d_transpose_2']['conv2d_transpose_2']['bias:0'].value]
model.layers[5].set_weights(weight)

weight = [group['conv2d_transpose_3']['conv2d_transpose_3']['kernel:0'].value,
          group['conv2d_transpose_3']['conv2d_transpose_3']['bias:0'].value]
model.layers[7].set_weights(weight)

weight = [group['conv2d_transpose_4']['conv2d_transpose_4']['kernel:0'].value,
          group['conv2d_transpose_4']['conv2d_transpose_4']['bias:0'].value]
model.layers[9].set_weights(weight)

weight = [group['conv2d_transpose_5']['conv2d_transpose_5']['kernel:0'].value,
          group['conv2d_transpose_5']['conv2d_transpose_5']['bias:0'].value]
model.layers[11].set_weights(weight)

weight = [group['conv2d_transpose_6']['conv2d_transpose_6']['kernel:0'].value,
          group['conv2d_transpose_6']['conv2d_transpose_6']['bias:0'].value]
model.layers[13].set_weights(weight)

weight = [group['conv2d_transpose_7']['conv2d_transpose_7']['kernel:0'].value,
          group['conv2d_transpose_7']['conv2d_transpose_7']['bias:0'].value]
model.layers[15].set_weights(weight)

weight = [group['conv2d_transpose_8']['conv2d_transpose_8']['kernel:0'].value,
          group['conv2d_transpose_8']['conv2d_transpose_8']['bias:0'].value]
model.layers[17].set_weights(weight)




enc = K.function([model.get_layer('encoder').input, K.learning_phase()],
				 [model.layers[-1].output])

print ("Loading Statistics...")
means = np.load('./model/means.npy')
stds  = np.load('./model/stds.npy')
evals = np.sqrt(np.load('./model/evals.npy'))
evecs = np.load('./model/evecs.npy')

sort_inds = np.argsort(-evals)
evals = evals[sort_inds]
evecs = evecs[:,sort_inds]

#Open a window
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((window_w, window_h))
face_surface_mini = pygame.Surface((input_w, input_h))
face_surface = screen.subsurface((drawing_x, drawing_y, drawing_w, drawing_h))
pygame.display.set_caption('Face Editor - By <CodeParade>')
font = pygame.font.SysFont("monospace", 15)

def update_mouse_click(mouse_pos):
	global cur_slider_ix
	global mouse_pressed
	x = (mouse_pos[0] - sliders_x)
	y = (mouse_pos[1] - sliders_y)

	if x >= 0 and y >= 0 and x < sliders_w and y < sliders_h:
		slider_ix_w = (x // slider_w)
		slider_ix_h = (y // slider_h)

		cur_slider_ix = slider_ix_h * slider_cols + slider_ix_w
		mouse_pressed = True

def update_mouse_move(mouse_pos):
	global needs_update
	y = (mouse_pos[1] - sliders_y)

	if y >= 0 and y < sliders_h:
		slider_row_ix = cur_slider_ix // slider_cols
		slider_val = y - slider_row_ix * slider_h

		slider_val = min(max(slider_val, slider_py), slider_h - slider_py) - slider_py
		val = (float(slider_val) / (slider_h - slider_py*2) - 0.5) * 6.0
		cur_params[cur_slider_ix] = val
		
		needs_update = True

def draw_sliders():
	for i in range(num_params):
		row = i // slider_cols
		col = i % slider_cols
		x = sliders_x + col * slider_w
		y = sliders_y + row * slider_h

		cx = x + slider_w / 2
		cy_1 = y + slider_py
		cy_2 = y + slider_h - slider_py
		pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))
		
		py = y + int((cur_params[i] / 6.0 + 0.5) * (slider_h - slider_py*2)) + slider_py
		pygame.draw.circle(screen, slider_color, (int(cx), int(py)), int(slider_w/2) - slider_px)

		cx_1 = x + slider_px
		cx_2 = x + slider_w - slider_px
		for j in range(7):
			ly = y + slider_h/2 + (j-3)*(slider_h/7)
			pygame.draw.line(screen, slider_color, (cx_1, ly), (cx_2, ly))

def draw_face():
	pygame.surfarray.blit_array(face_surface_mini, np.transpose(cur_face, (1, 0, 2)))
	pygame.transform.scale(face_surface_mini, (drawing_w, drawing_h), face_surface)
	pygame.draw.rect(screen, (0,0,0), (drawing_x, drawing_y, drawing_w, drawing_h), 1)
	
#Main loop
running = True
while running:
	#Process events
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			break
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				prev_mouse_pos = pygame.mouse.get_pos()
				update_mouse_click(prev_mouse_pos)
				update_mouse_move(prev_mouse_pos)
			elif pygame.mouse.get_pressed()[2]:
				cur_params = np.zeros((num_params,), dtype=np.float32)
				needs_update = True
		elif event.type == pygame.MOUSEBUTTONUP:
			mouse_pressed = False
			prev_mouse_pos = None
		elif event.type == pygame.MOUSEMOTION and mouse_pressed:
			update_mouse_move(pygame.mouse.get_pos())
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -3.0, 3.0)
				needs_update = True

	#Check if we need an update
	if needs_update:
		x = means + np.dot(evecs, (cur_params * evals).T).T
		#x = means + stds * cur_params
		x = np.expand_dims(x, axis=0)
		y = enc([x, 0])[0][0]
		cur_face = (y * 255.0).astype(np.uint8)
		needs_update = False
	
	#Draw to the screen
	screen.fill(background_color)
	draw_face()
	draw_sliders()
	
	#Flip the screen buffer
	pygame.display.flip()
	pygame.time.wait(10)
