# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def f(idx):
    return idx + 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print('aaa')
    for i in range(10):
        print(i + f(i))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
