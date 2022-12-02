import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import main

features =['bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g']
activation_function = ['sigmoid', 'hyperbolic tan']

screen = tk.Tk()

# config the screen window
screen.geometry('600x350')
screen.resizable(False, False)
screen.title('Task 1')

# label
label_select_function = ttk.Label(text="Select activation function",font=('calibre',10, 'bold'))
label_select_function.pack(fill=tk.X, padx=5, pady=5)

selected_activationFunc = tk.StringVar()
activationFunc_cb = ttk.Combobox(screen, textvariable=selected_activationFunc)

activationFunc_cb['values'] = [i for i in activation_function]

activationFunc_cb['state'] = 'readonly'

activationFunc_cb.pack(fill=tk.X, padx=5, pady=5)


nodes_var = tk.StringVar()
nodes_label = ttk.Label(screen, text = 'Enter nodes', font=('calibre',10, 'bold'))
nodes_label.pack(fill=tk.X,padx=5, pady=5)

nodes_entry = tk.Entry(screen,textvariable = nodes_var, font=('calibre',10,'normal'))
nodes_entry.pack(fill=tk.X,padx=5, pady=5)

eta_var=tk.StringVar()
eat_label = ttk.Label(screen, text = 'Enter learning rate', font=('calibre',10, 'bold'))
eat_label.pack(fill=tk.X,padx=5, pady=5)


name_entry = tk.Entry(screen,textvariable = eta_var, font=('calibre',10,'normal'))
name_entry.pack(fill=tk.X,padx=5, pady=5)

epochs_var=tk.StringVar()
epochs_label = ttk.Label(screen, text = 'Enter number of epochs', font=('calibre',10, 'bold'))
epochs_label.pack(fill=tk.X,padx=5, pady=5)


epochs_entry = tk.Entry(screen,textvariable = epochs_var, font=('calibre',10,'normal'))
epochs_entry.pack(fill=tk.X,padx=5, pady=5)


Checkbutton1 = tk.IntVar()  
checkbox = ttk.Checkbutton(screen, text = "Bias", 
                      variable = Checkbutton1,
                      onvalue = 1,
                      offvalue = 0,
                      )
  
def solve():
    if selected_activationFunc.get() == '' or nodes_var.get() == '' or epochs_var.get() == '' or eta_var.get() == '':
        showinfo(
            title='error',
            message='error! please reassign the form'
        )
    else:
        activation = selected_activationFunc.get()
        nodes = (nodes_var.get().split(','))
        nodes = [eval(i) for i in nodes]
        epochs = int(epochs_var.get())
        eta = float(eta_var.get())
        bias = True if Checkbutton1.get() == 1 else False
        accuracy = main.run(activation, nodes, eta, epochs, bias)
        showinfo(
            title = 'accuracy',
            message = f'activation = {activation}, epochs = {epochs}, eta = {eta}, bias = {bias}, nodes = {nodes}, accuracy = {accuracy}' 
        )

checkbox.pack(padx=5, pady=5)

B = ttk.Button(screen,text='solve',command=solve)
B.pack()
screen.mainloop()