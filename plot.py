############################## Validation Loss Plot
val_loss = newmodel.history['val_loss']
loss = newmodel.history['loss']
epochs = range(1, len(val_loss) + 1)

plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
plt.plot(epochs, loss, 'm--', label='Training Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks([2 * i for i in range((len(epochs)//2)+1)])
plt.grid(True)
plt.show()


############################## Training Accuracy plot
val_acc = newmodel.history['val_acc']
acc = newmodel.history['acc']
epochs = range(1, len(val_loss) + 1)

plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
plt.plot(epochs, acc, 'm--', label='Training Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks([2 * i for i in range((len(epochs)//2)+1)])
plt.grid(True)
plt.show()
