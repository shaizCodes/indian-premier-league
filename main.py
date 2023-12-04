from linear_regression import *
from classification import *

def main():
    while True:
        try:
            choice = int(input("Available Machine Learning Methods:\n    (1). Regression,\n    (2). Classification\n    (0). Exit program\nEnter your choice: "))
            if choice==1: doRegression()
            elif choice==2: doClassification()
            elif choice==0: break
            else:
                print("Invalid input! No such option exists!")
        except Exception as error:
            print(f"ERROR: {error}\nPlease enter valid input!")
if __name__=='__main__':
    main()