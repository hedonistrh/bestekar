# My Music Generation Journey (Part 1)

## INTRODUCTION

Neural networks are widely used in different areas such as cancer detection, autonomous cars, recommendation systems. With the [_Andraj Karpathy_'s post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) which is about RNN, generative Deep Learning (DL) become popular among different areas. With this post, researcher mostly focus _text generation_ for fun. However as you can see in the comments, some researcher give idea about music generation with Deep Learning.[]() Also, we can see great idea in this area like [Google's Magenta](https://magenta.tensorflow.org) and [Aiva](http://www.aiva.ai) which is Luxembourg based startup for music generation. Especially, Aiva's musics are amazing and their contents are registered under the France and Luxembourg authors’ right society (SACEM).

 With this impression, I want to start my own journey to this area. And this blog-post explains my first step to this journey.

 


## LSTM

[Colah's post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) gives great insight about LSTM. Also, I will try to give information about LSTM.

Traditional neural networks can not remember past information. They can only process current information. As you can think, if you can not remember past information, probably you can not even make meaningful sentences. Recurrent Neural Network(RNN) solve this problem with recurrent connection via loops at nodes. However, Vanilla RNN has another problem called as _vanishing gradient_. At this point, you can ask what is gradient and why this problem is big deal. Let me explain these concepts in one paragraph.

Gradient is fancy words for slope of line a.k.a derivative. You can find the minimum or maximum point of the lines thanks to gradient. As you expect, ıts usage for Deep Learning comes with the loss function. Our aim is minimize the loss function for DL models. So that, when you want to find the point for the mimimum loss, you have to use gradient. Gradient based methods learn a parameter's value (weights of node or biases) by understanding how a small change in this parameter's value will affect the outputs of the network. When vanishing gradient problem occurs, gradient of early layers of the model's parameters' become very small. Thus, DL model can not find the better value for parameter effectively to decrease loss function with find the minimum point of line thanks to gradient.



## _.mid_ Files

_.mid_ files include _midi_ datas. _Midi_ means that _Musical Instrument Digital Interface_. 

This type of files do not include actual audio as opposed to _.mp3_ and _.wav_.  _.mid_ files explain what notes are played and how long or loud each note should be. 

## Framework

I have used _Keras_ as Deep Learning framework with _Tensorflow_ backend. Because, it is easier than pure _Tensorflow API_ for me.


## Code Part

When you want to feed your deep learning model, you need input as matrix format. So that, I should convert _.midi_ files to matrix format. For this process
- Read midi file and extract information about notes, durations and offsets.
- Convert these informations to matrix.


NOTE: You can find full code in [my _GitHub_ repository](https://github.com/hedonistrh/bestekar). I will provide snippets from code for better understanding.

**Let's Start!**

- Firstly, we need extract _information_ from _.mid_ file. We need three information which are notes, duration of notes and offset of notes. (_Duration_ represent the how long played, _offset_ represent the when played) Also, I need just piano part's information. 

```python
midi = music21.converter.parse(filename)
notes_to_parse = None

parts = music21.instrument.partitionByInstrument(midi)

instrument_names = []

try:
    for instrument in parts: # Learn names of instruments.
        name = (str(instrument).split(' ')[-1])[:-1]
        instrument_names.append(name)

except TypeError:
    print ('Type is not iterable.')
    return None
    
    # Just take piano part. For the future works, we can use different instrument.
    try:
        piano_index = instrument_names.index('Piano')
    except ValueError:
        print ('%s have not any Piano part' %(filename))
        return None
    
    
    notes_to_parse = parts.parts[piano_index].recurse()
    
    duration_piano = float(check_float((str(notes_to_parse._getDuration()).split(' ')[-1])[:-1]))

    durations = []
    notes = []
    offsets = []
    
    for element in notes_to_parse:
        if isinstance(element, note.Note): # If it is single note
            notes.append(note_to_int(str(element.pitch))) # Append note's integer value to "notes" list.
            duration = str(element.duration)[27:-1] 
            durations.append(check_float(duration)) 
            offsets.append(element.offset)

        elif isinstance(element, chord.Chord): # If it is chord
            notes.append('.'.join(str(note_to_int(str(n)))
                                  for n in element.pitches))
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

```

Now we have three different list which are for 
- Notes 
- Duration
- Offset

Note: I have convert note's representation from letter to integer. This process has done by *note_to_int* function.

``` python
def note_to_int(note): # converts the note's letter to pitch value which is integer form.
 
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if ('#-' in note):
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[3]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('#' in note): # not totally sure, source: 
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('-' in note): 
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    else:
        first_letter = note[0]
        base_val = note_base_name.index(first_letter)
        octave = note[1]
        value = base_val + 12*(int(octave)-(-1))
        
    return value
```


Create matrix to represent midi file with using information which comes from previous lists.

- Create matrix with random uniform values. 
    - X-axis of this matrix will represent time (duration and offset) and Y-axis will represent the frequency (pitch a.k.a notes). This matrix will be like spectogram. 
    - We have 128 different pitch value, so that length of matrix's Y-axis will be equal to _128_. For the time representation, I have choosen _0.25_ note length as minimum value. Because, most notes is multiplication of _0.25_. 

    ```python
    try:
        last_offset = int(offsets[-1]) 
    except IndexError:
        print ('Index Error')
        return (None, None, None)
    
    total_offset_axis = last_offset * 4 + (8 * 4) 
    our_matrix = np.random.uniform(min_value, lower_first, (128, int(total_offset_axis))) 
    ```
- Read lists and extract information to modify matrix to represent midi. I have spent too much time to determine how distunguish between a long note and many short notes. According to my trials, best method is represent a long note with bigger value at first occurence's offset, smaller value at continuation's offset. For instance, **C4** with duration _0.75_ will be represented as _1.0-0.5-0.5_, three _0.25_ **C4** will be represented as _1.0-1.0-1.0_. However, for better generalization, we can add some randomness to these values. But, in this codes, I have not done that. (If you want to this, for instance you can change lower_first to 0.1, lower_second to 0.4, upper_first to 0.6, upper_second to 0.8)

    ```python
    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration)/0.25) # indicates time duration for single note.
       
        # Define difference between single and double note.
        # I have choose the value for first touch, the another value for continuation.
        # Lets make it randomize
        
        # I choose to use uniform distrubition. Maybe, you can use another distrubition like Gaussian.
         
        first_touch = np.random.uniform(upper_second, max_value, 1)
        continuation = np.random.uniform(lower_second, upper_first, 1)
        
        if ('.' not in str(note)): # It is not chord. Single note.
            our_matrix[note, int(offset * 4)] = first_touch
            our_matrix[note, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation

        else: # For chord
            chord_notes_str = [note for note in note.split('.')] 
            chord_notes_float = list(map(int, chord_notes_str)) # Take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, int(offset * 4)] = first_touch
                our_matrix[chord_note_float, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation
                
    ```


Now, we can create dataset from _.mid_ file with this functions. For the this post, I have used great composer Schumann _.mid_ files. Firstly, I have convert all midi file to matrix one by one and append single midi's matrix to list of cumulative matrix. After that, I have convert list to numpy array and save this array which includes Schumann's _.mid_ to _.npy_ file to use easily with another platform and later use.

``` python
# Build database

database_npy = 'midis_array_schumann.npy'
my_file_database_npy = Path("./database_npy/" + database_npy )


if my_file_database_npy.is_file(): 
    midis_array = np.load(my_file_database_npy)
    
else:
    print (os.getcwd())
    root_dir = ('./midi_files')
    all_midi_paths = glob.glob(os.path.join(root_dir,'classic/schumann/*mid'))
    print (all_midi_paths)
    matrix_of_all_midis = []

    # All midi have to be in same shape. 

    for single_midi_path in all_midi_paths:
        print (single_midi_path)
        matrix_of_single_midi = midi_to_matrix(single_midi_path, length=250)
        if (matrix_of_single_midi is not None):
            matrix_of_all_midis.append(matrix_of_single_midi)
            print (matrix_of_single_midi.shape)
    midis_array = np.asarray(matrix_of_all_midis)
    np.save(my_file_database_npy, midis_array)
```

When you load _.npy_ file to numpy array, your array's shape will be (# of file, # of freq, # of time in a single file). You can not use this type of array directly. So that, we have to modify this data to use with _LSTM._ 

- Firstly, I will convert to (# of file, # of time in a single file, # of freq)

```python
midis_array = np.transpose(midis_array_raw, (0, 2, 1)) 
midis_array = np.asarray(midis_array)
```

- Secondly, convert to (# of freq, # of file * # of time in a single file)

```python
midis_array = np.reshape(midis_array,(-1,128))
midis_array.shape
```

- Finally, create 2 different array for training. First one will be used to predict next array, and second one will represent true array. Weights of layer of LSTM will be based on error between prediction array which is based on first array and true array. With gradient descent, model update each layer to decrease this error.

```python
max_len = 18 # how many column will take account to predict next column.
step = 1 # step size.

# ADD ILLUSTRATION ABOUT THIS PROCESS.

previous_full = []
predicted_full = []

for i in range (0, midis_array.shape[0]-max_len, step):
    prev = midis_array[i:i+max_len,...] # take max_len column.
    pred = midis_array[i+max_len,...] # take (max_len)th column.
    previous_full.append(prev)
    predicted_full.append(pred)
```

Now we can build our deep learning model with _KERAS_ Api. I have used 3 LSTM layer and 2 Dense Layer. Also, model's final activation function is _softmax_ because we want to output which is between 0 and 1. Also, I have used _LeakyReLU_ and _Dropout_ layers to get rid of gradient problem.

```python
# Build our Deep Learning Architecture

from keras import layers
from keras import models
import keras
from keras.models import Model
import tensorflow as tf
from keras.layers.advanced_activations import *

midi_shape = (max_len, 128)

input_midi = keras.Input(midi_shape)

x = layers.LSTM(512, return_sequences=True)(input_midi)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization() (x)
x = layers.Dropout(0.4)(x)

x = layers.LSTM(512, return_sequences=True)(input_midi)
x = layers.LeakyReLU()(x)
# x = layers.BatchNormalization() (x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256)(x)
x = layers.LeakyReLU()(x)
# x = layers.BatchNormalization() (x)
x = layers.Dropout(0.4)(x)

x = layers.LSTM(128)(x)
x = layers.LeakyReLU()(x)
# x = layers.BatchNormalization() (x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation='softmax')(x) # Maybe, we can use sigmoid.

model = Model(input_midi, x)
```

We should compile this model. So that, we need tune two things.
- Optimizer
- Loss Function

I have used Stochastic Gradient Descent (SGD) for optimizer and _categorical cross entropy_ for loss function.

``` python
optimizer = keras.optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

Now, when we feed our deep learning model with training data, it predict same values. We use these values to sample. This part is a litte bit confusing. I have tried different methods. For instance, assume argument of max value as first touch and second max argument as continuation, however, it can not creates enjoyable music. So that, I have used this function.

``` python
def sample_v2(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    num_of_top = 10
    num_of_first = np.random.randint(1,3)

    ind = np.argpartition(preds, -1*num_of_top)[-1*num_of_top:]
    top_indices_sorted = ind[np.argsort(preds[ind])]
    
    array = np.random.uniform(0.0, 0.0, (128)) 
    array[top_indices_sorted[0:num_of_first]] = 1.0
    array[top_indices_sorted[num_of_first:num_of_first+4]] = 0.5
    
    return array
```

Now we can train our system and generate array for converting to midi.

``` python
import random
import sys

epoch_total = 81
batch_size = 2

for epoch in range(1, epoch_total): 
    print('Epoch:', epoch)
    model.fit(previous_full, predicted_full, batch_size=batch_size, epochs=1,
              shuffle=True)
    
    start_index = random.randint(0, len(midis_array)- max_len - 1)
    
    generated_midi = midis_array[start_index: start_index + max_len]
    
    
    if ((epoch%10) == 0):
      model.save_weights('my_model_weights.h5')

      for temperature in [1.2]:
          print('------ temperature:', temperature)
          # sys.stdout.write(generated_text)

          for i in range(480):
              samples = generated_midi[i:]
              expanded_samples = np.expand_dims(samples, axis=0)
              preds = model.predict(expanded_samples, verbose=0)[0]
              preds = np.asarray(preds).astype('float64')

              next_array = sample_v2(preds, temperature)
              
              midi_list = []
              midi_list.append(generated_midi)
              midi_list.append(next_array)
              generated_midi = np.vstack(midi_list)
              
          generated_midi_final = np.transpose(generated_midi,(1,0))
          output_notes = matrix_to_midi(generated_midi_final, random=0)
          midi_stream = stream.Stream(output_notes)
          midi_stream.write('midi', fp='lstm_output_v1_{}_{}.mid'.format(epoch, temperature))
``` 

We need *matrix_to_midi* function for upper script. Now, let's build it!

- First read the matrix. (I have provide small part of codes. Because codes are too long to read in blogpost)

``` python
for y_axis_num in range(y_axis):
        one_freq_interval = matrix[y_axis_num,:] # values in one column
        one_freq_interval_norm = converter_func(one_freq_interval)
        # print (one_freq_interval)
        i = 0        
        offset = 0
        while (i < len(one_freq_interval)):
            how_many_repetitive = 0
            temp_i = i
            if (one_freq_interval_norm[i] == first_touch):
                how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i+1, continuation=continuation)
                i += how_many_repetitive 
            if (how_many_repetitive > 0):
                new_note = note.Note(int_to_note(y_axis_num),duration=duration.Duration(0.25*how_many_repetitive))
                new_note.offset = 0.25*temp_i
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            else:
                i += 1
        
    return output_notes
```

As you can see, there is some function in this code. Now, look these functions.

- *Converter_func* is for give unique numbers to represent first touch, continuation and rest. If you use range for represent these values at *midi_to_matrix* you need this function.

```python
def converter_func(arr,first_touch = 1.0, continuation = 0.0, lower_bound = lower_bound, upper_bound = upper_bound):
    # first touch represent start for note, continuation represent continuation for first touch, 0 represent end or rest
    np.place(arr, arr < lower_bound, -1.0)
    np.place(arr, (lower_bound <= arr) & (arr < upper_bound), 0.0)
    np.place(arr, arr >= upper_bound, 1.0)
    return arr
```
- *how_many_repetitive_func* is used to understand duration of note. 
```python
def how_many_repetitive_func(array, from_where=0, continuation=0.0):
    new_array = array[from_where:]
    count_repetitive = 1 
    for i in new_array:
        if (i != continuation):
            return (count_repetitive)
        else:
            count_repetitive += 1
    return (count_repetitive)
```

## Results

Let's listen some outputs of the system9