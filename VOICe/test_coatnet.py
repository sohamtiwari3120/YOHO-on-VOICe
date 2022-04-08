from audioop import add
from config import add_EAP_to_path
add_EAP_to_path()
from model.attention.CoAtNet import CoAtNet as CN
def main():
    model = CN(3, 257)
    print(model)

if __name__ == '__main__':
    main()