
class node:
    def __init__(self,data):
        self.data = data
        self.Next = None


def insert(head,data):
    temp = head
    while temp.Next:
        temp = temp.Next
    temp.Next = node(data)
    return head

def disp(head):
    temp = head
    while temp:
        print(temp.data,end=" ")
        temp = temp.Next

def reverse(head):
    temp1 = head
    temp2 = head.Next
    temp1.Next = None
    while temp2:
        temp3 = temp2.Next
        temp2.Next = temp1
        temp1 = temp2
        temp2 = temp3
    return temp1

head = node(0)

for i in range(1,9):
    head = insert(head,i)
disp(head)
head2 = head
for i in range(4):
    head2 = head2.Next
temp1 = head2
head2 = head2.Next
temp1.Next = None

head2 = reverse(head2)
print()
disp(head)
print()
disp(head2)
print()

temp1 = head
temp2 = head2
while temp1 and temp2:
    temp3 = temp1.Next
    temp4 = temp2.Next

    temp1.Next = temp2
    temp2.Next = temp3

    temp1 = temp3
    temp2 = temp4
# if temp2:
#     temp1.Next = temp2
disp(head)