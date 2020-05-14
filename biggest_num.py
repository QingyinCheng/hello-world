 #Cassie Cheng
 #CS021
 #Display three numbers in descending order

#Get the three numbers
num1 = int(input('First number?: '))
num2 = int(input('Second number?: '))
num3 = int(input('Third number?: '))

#Get the biggest number
if (num1 > num2) and (num1 > num3):
    largest=num1
elif (num2 > num1) and (num2 > num3):
    largest=num2
else:
    largest=num3

#Get the smallest number
if (num1 < num2) and (num1 < num3):
    smallest=num1
elif (num2 < num1) and (num2 < num3):
    smallest=num2
else:
    smallest=num3

#Get the middle number
if (num1!=largest) and (num1!=smallest):
    middle=num1
elif (num2!=largest) and (num2!= smallest):
    middle=num2
else:middle=num3


#get the order of three numbers
print("Desecending order:",largest, middle, smallest)

