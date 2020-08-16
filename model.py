
# We have 2 inputs, 1 for each picture
left_input = Input((160,80,3))
right_input = Input((160,80,3))

# We will use 2 instances of 1 network for this task
conv_base = efn.EfficientNetB0(weights='imagenet',
                  include_top=False,
                  input_shape=(160,80,3))

for layer in conv_base.layers[:5]:
    layer.trainable = False
out = conv_base.output

out=Conv2D(4096,(1,1),padding="same",activation="relu")(out)
convnet = Model(conv_base.input, output=out)
convnet.name="EfficientNetB1"

# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

#Identification Prediction
y=Flatten()(encoded_l)

z=Flatten()(encoded_r)


# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.square(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([y, z])


L1_distance = Dropout(rate=0.2)(L1_distance)
sigmoid = Dense(1,activation='sigmoid',name='bclss')(L1_distance)

siamese_net = Model(inputs=[left_input,right_input],outputs=sigmoid)


rs = optimizers.RMSprop(lr=1e-4)
siamese_net.compile(loss={'bclss':'binary_crossentropy'},
                    optimizer=rs, metrics={'bclss':'accuracy'})


#checkpoint = ModelCheckpoint('/content/models/model-{epoch:03d}.h5', verbose=1, save_weights_only=False, mode='auto')
#callbacks=[checkpoint]
start_time = datetime.datetime.now()
newmodel=siamese_net.fit([left_train,right_train], {'bclss':targets}, 
                         batch_size=128, 
                         epochs=18, 
                         verbose=1, validation_data=([left_val,right_val],{'bclss':val_targets}))
end_time = datetime.datetime.now()
print ('* total training time:', str(end_time-start_time))
#shuffle=True
#siamese_net.save_weights('drive/My Drive/thesis/new change parametr/weight/model-{epoch:03d}.h5',overwrite=False)