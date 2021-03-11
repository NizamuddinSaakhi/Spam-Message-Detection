from tkinter import *
from nltk.corpus import stopwords
from NBmodel import spam_detect_model, cv
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.tokenize import word_tokenize

root = Tk()

root.title('Spam Message Detection')
root.iconbitmap('spam.ico')

temp_var = StringVar()

def deleteLabel():
    myLabel1.pack_forget()

def submit():
    global myLabel1
    text = input.get()
    def singleFunction():
        def singleMessage(text):
            stop_words = set(stopwords.words('english'))
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = word_tokenize(text)
            text = [w for w in text if not w in stop_words]
            text = [ps.stem(word) for word in text]
            text = " ".join(text)
            return text

        ls=[]
        ls.append(singleMessage(text))
        sp = cv.transform(ls).toarray()
        y_pred2 = spam_detect_model.predict(sp)
        if y_pred2 == 1:
            return "spam message"
        else:
            return "normal message"

    temp = "This is " + singleFunction()
    myLabel1 = Label(root, text=temp, font=("Times New Roman", 20))
    myLabel1.pack(pady=10)

myLabel2 = Label(root, text="Spam Message Detection", font=("Times New Roman", 25) ,pady=25)
myLabel2.pack()

myLabel = Label(root, text="Enter message to analyse", font=("Times New Roman", 20))
myLabel.pack()
input = Entry(root,width=50, borderwidth=2, textvariable=temp_var, font=("Times New Roman", 20))
input.pack()

my_button = Button(root, text="Click to Analyse", command = submit, font=("Times New Roman", 15))
my_button.pack(pady=10)

reset_button = Button(root, text = "Reset" , command = deleteLabel, font=("Times New Roman", 15))
reset_button.pack(pady=10)

quitButton = Button(root, text="Exit", command=root.quit, font=('Times New Roman', 15))
quitButton.pack()

root.mainloop()