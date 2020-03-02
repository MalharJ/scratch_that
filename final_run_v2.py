import torchvision.models as models
from torchsummary import summary

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
device = torch.device("cuda")
import pretrainedmodels
from PIL import Image
import requests
import time
import pretrainedmodels.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES
from skimage.draw import line_aa
from skimage.draw import circle
import torchvision
from random import randint
print (device)

###############################

LABELS_URL = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json'
IMG_URL = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/cat.jpg'

labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

model_names = ['resnet18', 'squeezenet1_0', 'squeezenet1_1', 'resnet50', 'vgg11_bn', 'vgg19', 'vgg16', 'resnet34', 'vgg13', 
           'densenet201', 'vgg13_bn', 'resnet152', 'vgg16_bn', 'resnet101', 
               'vgg19_bn', 'vgg11', 'alexnet', 'inceptionv3']

def visualize_torch(img):
    plt.figure()
    plt.imshow(torchvision.utils.make_grid(img, nrow=5).permute(1, 2, 0))
    
def torch_softmax(predictions):
    m = torch.nn.Softmax(dim=0)
    return m(predictions)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def top_n_indices(arr, n):
    tmp = arr.argsort()[-1*n:][::-1]
    return list(tmp)

def prep_image_for_inference(img_path):
    input_img = load_img(img_path)
    input_tensor = tf_img(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    input_to_model = torch.autograd.Variable(input_tensor,
            requires_grad=False)
#     input_to_model -= torch.min(input_to_model)
#     input_to_model = input_to_model / torch.max(input_to_model)
    return input_to_model

def get_probabilities(logits):
    logits_numpy = logits.detach().numpy()
    return softmax(logits_numpy[0])

def torch_image_to_numpy(img):
    input_img = img.detach().numpy()
    return input_img

def prep_torch_img_for_viewing(img):
    return torchvision.utils.make_grid(img, nrow=5).permute(1, 2, 0)

def prep_numpy_img_for_viewing(img):
    img = img[0,:,:,:]
    print (img.shape)
    img = np.reshape(img, [299, 299, 3])
    return img

def numpy_img_from_path(img_path):
    input_img = load_img(img_path)
    return np.asarray(input_img)

model_name = 'vgg19'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
model.to(device)

# Load in the filenames 
import os
path = '/home/malhar/Desktop/data/'

imagenet_validation_filenames_path = path + 'imagenet_validation_filenames.txt'
imagenet_validation_sys_labels_path = path + 'imagenet_2012_validation_synset_labels.txt'
imagenet_validation_word_labels_path = path + 'synset_labels_to_words.txt'

imagenet_validation_filenames = []
imagenet_validation_sys_labels = []
imagenet_validation_word_labels = []



with open(imagenet_validation_filenames_path) as f:
    for line in f:
        imagenet_validation_filenames.append(line.rstrip('\n'))
        
with open(imagenet_validation_sys_labels_path) as f:
    for line in f:
        imagenet_validation_sys_labels.append(line.rstrip('\n'))
                
with open(imagenet_validation_word_labels_path) as f:
    for line in f:
        imagenet_validation_word_labels.append(line.rstrip('\n'))
        
import random

RANGE = 50000
NUMBER_OF_FILES = 50
random_filename_numbers = random.sample(xrange(RANGE), NUMBER_OF_FILES)

# Get the randomly chosen files
random_filenames = [imagenet_validation_filenames[i] for i in random_filename_numbers]
corresponding_sysnet_labels = [imagenet_validation_sys_labels[i] for i in random_filename_numbers]
corresponding_index_labels = []

for sysnet_label in corresponding_sysnet_labels:
    index = [idx for idx, s in enumerate(imagenet_validation_word_labels) if sysnet_label in s][0]
    corresponding_index_labels.append(index)


# Define the fitness function
# from https://github.com/CMA-ES/pycma/blob/master/cma/fitness_functions.py
def custom_fitness_function(model, img, original_class, target_class, rr, cc, solution_vector, option):
    
    size = len(solution_vector)
    
    r_vector = torch.from_numpy(solution_vector[0:size/3])
    r_vector = r_vector.type(torch.float)
    r_vector = r_vector.to(device)
    
    g_vector = torch.from_numpy(solution_vector[size/3: int(2*size/3)])
    g_vector = g_vector.type(torch.float)
    g_vector = g_vector.to(device)
    
    b_vector = torch.from_numpy(solution_vector[int(2*size/3): size])
    b_vector = b_vector.type(torch.float)
    b_vector = b_vector.to(device)
    
    img[0,0,rr,cc] = r_vector
    img[0,1,rr,cc] = g_vector
    img[0,2,rr,cc] = b_vector
    
    prediction = model(img)
    probabilities = torch_softmax(prediction[0])
    
    # Option 1 aka trash
    # NVM it actually isnt too bad; just less descendants per generation and more generations. Does't converge to 
    # what we want though
    if (option == 0):
        # Get the target class probability - we want to maximize this
        target_class_prob = torch.log(probabilities[target_class])
        
        # Get the probability vectors in numpy format
        numpy_probs = probabilities.detach().cpu().numpy()
        
        # Get the top k indices that have the max probabilities
        top_k = (top_n_indices(numpy_probs, 5))
        
        top_k_probs = probabilities[top_k]
        
        # If the target class is in the top k, we need to take some precautions. We need to set that value to 0
        if (int(target_class) in top_k):
            
            # Find the index where the target class is in the top k
            index = top_k.index(target_class)

            # Set that value to zero
            top_k_probs[index] = 0
            
        
        original_class_prob_others = (torch.sum(torch.log(top_k_probs)))    
        original_class_prob = torch.log(probabilities[original_class])
        fitness = 10 *(target_class_prob)  - (original_class_prob)

	if (fitness > 0):
		fitness = 10*target_class_prob - original_class_prob_others
    
        
    # Maximize target class probability and minimize non-target class probability; squeeze all probability into target
    elif (option == 1):
        target_class_prob = torch.log(probabilities[target_class])
        
        top_scorer_index = int(torch.argmax(probabilities))
        
        top_scorer_loss = 0
        
        if (top_scorer_index != target_class):
            top_scorer_loss = torch.log(probabilities[top_scorer_index])
        
        everything_log = probabilities
        everything_log[target_class] = 1
        everything_log = torch.log(everything_log)
        non_target_class_fitness = torch.sum(everything_log)
        fitness = target_class_prob - non_target_class_fitness #- top_scorer_loss
    
    elif (option == 2):
        target_class_prob = torch.log(probabilities[target_class])
        everything_log = probabilities
        everything_log[target_class] = 1
        everything_log = torch.log(everything_log)
        non_target_class_fitness = torch.sum(everything_log)
        entropy = torch.dot(torch.log(probabilities), probabilities)
        fitness = target_class_prob - non_target_class_fitness - entropy
    
    return fitness

# fit_func = rastrigin
fit_func = custom_fitness_function

def get_predictions_from_temp_array(result, original_class, target_class, rr, cc):
    adversarial_scratch = result[0]
    
    size = len(adversarial_scratch)
    
    r_vector = torch.from_numpy(adversarial_scratch[0:size/3])
    r_vector = r_vector.type(torch.float)
    r_vector = r_vector.to(device)

    g_vector = torch.from_numpy(adversarial_scratch[size/3: int(2*size/3)])
    g_vector = g_vector.type(torch.float)
    g_vector = g_vector.to(device)

    b_vector = torch.from_numpy(adversarial_scratch[int(2*size/3): size]).to(device)
    b_vector = b_vector.type(torch.float)
    b_vector = b_vector.to(device)
    
    input_to_model_temp = prep_image_for_inference(path_img)
    input_to_model_temp = input_to_model_temp.to(device)
    input_to_model_temp[0,0,rr,cc] = r_vector
    input_to_model_temp[0,1,rr,cc] = g_vector
    input_to_model_temp[0,2,rr,cc] = b_vector
    
    initial_logits_tmp = model(input_to_model_temp)

    initial_probabilities_tmp = get_probabilities(initial_logits_tmp.cpu())
    
    return initial_probabilities_tmp[original_class], initial_probabilities_tmp[target_class], initial_probabilities_tmp


# defines a function to use solver to solve fit_func
def genetic_solver(x0,x1,y0,y1,solver, input_image, original_class, target_class, option, input_file,img_name):
    # Have the histories vector
    history = []
    orig_probs = []
    target_probs = []
    
    probabilities_all = []
    
    # Go through each generation
    for j in range(MAX_ITERATION):
        # Generate the solutions
        solutions = solver.ask()
        
        # Calculate the fitness
        fitness_list = np.zeros(solver.popsize)
        
        # Calculate the fitnesses
        for i in range(solver.popsize):
            fitness_list[i] = custom_fitness_function(model, input_to_model, original_class, target_class, rr, cc, solutions[i], option)
        
	if (np.isnan(fitness_list).any() or np.isinf(fitness_list).any()):
		print ('RIP ', j)
		continue
        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        
        orig_prob, target_prob, probabilities = get_predictions_from_temp_array(result, original_class, 
                                                                                target_class, rr, cc)
        
        if (target_prob == np.max(probabilities)):
            
            input_file.write("Reached max! Iteration: " +str(j) + " probability:: " +str(target_prob))
            input_file.write("\n")
            input_file.write("Top scoring class: " + str(labels[int(np.argmax(probabilities))]))
            input_file.write("\n")
            
            import pickle
            pickle_filename = 'pickled_vgg19/' + img_name + '_target=' + str(target_class)
            pickle_filename += '_original=' + str(original_class) +'_x0_' + str(x0) + '_y0_' + str(y0) + '_x1_' + str(x1) + '_y1_' + str(y1) + '.p'
            pickle.dump( result[0], open( pickle_filename, "wb" ) )
            
            return j,history, probabilities_all, orig_probs, target_probs
        
        history.append(result[1])
        orig_probs.append(orig_prob)
        target_probs.append(target_prob)
        probabilities_all.append(probabilities)
        
        if (j+1)%2 == 0:
            
            input_file.write("fitness at iteration " + str(j+1) + ' ' + str(result[1]))
            input_file.write("\n")
            input_file.write("Target probability: " +str(target_prob) + ":: Original probability: " +str(orig_prob))
            input_file.write("\n")
            input_file.write('Top scorer: '+ str(labels[int(np.argmax(probabilities))]) +
                   ', probability:: ' + str(np.max(probabilities)))
            input_file.write("\n")

    return j,history, probabilities_all, orig_probs, target_probs


load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)

NUMBER_OF_LINE_PERMUTATIONS = 10

import random
for iterations in range(NUMBER_OF_FILES):
    target_class = random.randint(0,1000)    
    for permutation in range(NUMBER_OF_LINE_PERMUTATIONS):

        xpoint = random.randint(0,224)
        ypoint = random.randint(0,224)
        xpoint2 = random.randint(0,224)
        ypoint2 = random.randint(0,224)
    #     xpoint,ypoint,xpoint2,ypoint2 = 13,5,200,210
        print ('Random points: ', xpoint,ypoint,xpoint2,ypoint2)

        # Prep image for inference - load it in and do a bunch of stuff so its ready
        path_img = path + 'imagenet_validation_images/' + random_filenames[iterations]
        input_to_model = prep_image_for_inference(path_img)
        input_to_model = input_to_model.to(device)
        
        rr, cc, val = line_aa(xpoint, ypoint,xpoint2, ypoint2)
        print (len(rr))

        # Get the adversarial line to extract
        adversarial_scratch = np.zeros(1)
        try:
            adversarial_scratch = input_to_model[:,:,rr,cc]
        except:
            print ('what')
        # Define the constants we want

	print (adversarial_scratch.shape)

	if (adversarial_scratch is not None and len(adversarial_scratch.shape) == 3):
        	NPARAMS = adversarial_scratch.shape[1] * adversarial_scratch.shape[2]        # make this a 100-dimensinal problem.
        	NPOPULATION = 80    # use population size of 200.
        	MAX_ITERATION = 5000 # run each solver for 4000 generations.
        	import numpy as np

        	fitness_option = 0
        	original_class = corresponding_index_labels[iterations]

       		filename = 'log_vgg19/imageName_' + random_filenames[iterations]
        	filename += '_fitness_' + str(fitness_option)
        	filename += '_target=' + str(target_class) 
        	filename += '_original=' + str(original_class)
        	filename += '_npop=' + str(NPOPULATION)
        	filename += '_iters=' + str(MAX_ITERATION)
        	filename += 'range_limit'
        	filename += '_x0=' + str(xpoint)
        	filename += '_y0=' + str(ypoint)
        	filename += '_x1=' + str(xpoint2)
        	filename += '_y1=' + str(ypoint2)
        	log_file= open(filename,"w")

        	# defines CMA-ES algorithm solver for each target class
        	cmaes = CMAES(NPARAMS,
                	popsize=NPOPULATION,
                	sigma_init = 0.5
              		)

        # Solve a k dimensional problem, and choose 4000 descendants at each iteration
        	iters, scratch, probabilities_all, orig_probs, target_probs = genetic_solver(xpoint,xpoint2,ypoint,ypoint2,cmaes, input_to_model, original_class,target_class, fitness_option, log_file, random_filenames[iterations])
		print ('Iterations taken: ', iters)
        	log_file.write('Iterations taken: ' + str(iters))
